//! Custom axum extractors that close pitfalls in axum's defaults.
//!
//! ## Why
//!
//! axum's stock `Option<Json<T>>` extractor has a behaviour that is
//! easy to overlook: if the request body is present but malformed
//! (invalid JSON, hits the deserializer's recursion limit, exceeds
//! the body limit, etc.) the extractor **silently returns `None`**
//! instead of producing a `JsonRejection`. The handler then sees
//! `body.is_none()` and typically falls back to defaults.
//!
//! The 2026-04-25 OSS code review verified this empirically: a
//! 70-level deep nested-bool POST against `/idx/_search` arrived at
//! `parse_request` as `{"query": {"match_all": {}}}` — the deep query
//! had been silently replaced with the default. The user got back
//! 200 OK with empty hits, never knowing the query they sent had
//! been thrown away.
//!
//! `OptionalJson<T>` distinguishes the three real cases:
//!
//! | Case               | Behaviour                            |
//! |--------------------|--------------------------------------|
//! | Body absent/empty  | `Ok(OptionalJson(None))`             |
//! | Body parses to `T` | `Ok(OptionalJson(Some(T)))`          |
//! | Body is malformed  | `Err(JsonRejection)` → 400 response  |
//!
//! Use this on every POST/PUT handler that accepts a body but
//! tolerates an absent one (e.g. `/_search` works with no body —
//! defaults to match-all). For mandatory-body endpoints, use
//! `Json<T>` directly.
//!
//! ## Migration
//!
//! ```ignore
//! // Before:
//! pub async fn handler(body: Option<Json<EsSearchBody>>) {
//!     let body = body.map(|j| j.0).unwrap_or_default();
//! }
//!
//! // After:
//! use crate::extract::OptionalJson;
//! pub async fn handler(body: OptionalJson<EsSearchBody>) {
//!     let body = body.0.unwrap_or_default();
//! }
//! ```

use axum::{
    async_trait,
    body::Bytes,
    extract::{rejection::BytesRejection, FromRequest, Request},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::de::DeserializeOwned;

/// JSON-body extractor that returns `None` when the body is absent
/// but rejects (rather than silently defaulting) when the body is
/// present and invalid. See module docs for rationale.
pub struct OptionalJson<T>(pub Option<T>);

/// Why an `OptionalJson<T>` extraction failed. Always converts to a
/// 400 with an ES-shaped error body so clients see the real reason
/// instead of getting a misleading 200 with default behaviour.
#[derive(Debug)]
pub enum OptionalJsonRejection {
    /// Reading the body bytes failed (Content-Length oversize,
    /// connection reset, etc.). Forwarded from axum's `Bytes`
    /// extractor.
    BodyRead(BytesRejection),
    /// The body bytes did not parse as JSON for `T`.
    JsonParse(serde_json::Error),
    /// `Content-Type` header was set to something other than a JSON
    /// flavour *and* the body did not parse as JSON for `T`. Most clients
    /// send `application/json`; `application/x-ndjson` is accepted for the
    /// bulk endpoints (those use their own extractor anyway, so this only
    /// fires for misuse). A well-formed JSON body is accepted whatever the
    /// header says — `curl -d` sends `application/x-www-form-urlencoded`,
    /// and rejecting that body broke every documented `curl -d` recipe
    /// (#2575).
    UnsupportedMediaType(String),
}

impl IntoResponse for OptionalJsonRejection {
    fn into_response(self) -> Response {
        let (status, reason) = match self {
            OptionalJsonRejection::BodyRead(r) => {
                (r.status(), format!("failed to read request body: {r}"))
            }
            OptionalJsonRejection::JsonParse(e) => (
                StatusCode::BAD_REQUEST,
                format!("malformed JSON in request body: {e}"),
            ),
            OptionalJsonRejection::UnsupportedMediaType(ct) => (
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                format!(
                    "unsupported Content-Type: {ct}; this endpoint needs a JSON body, so send \
                     Content-Type: application/json (or omit the header). The body was not \
                     valid JSON either"
                ),
            ),
        };
        let body = serde_json::json!({
            "error": {
                "type": "parse_exception",
                "reason": reason,
            },
            "status": status.as_u16(),
        });
        (status, Json(body)).into_response()
    }
}

#[async_trait]
impl<S, T> FromRequest<S> for OptionalJson<T>
where
    S: Send + Sync,
    T: DeserializeOwned,
{
    type Rejection = OptionalJsonRejection;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // Tolerate a missing or generic Content-Type for empty bodies, and
        // let a well-formed JSON body through whatever the header says (see
        // below). A non-JSON header is only named when the body failed to
        // parse, which is the case it actually explains.
        let ct = req
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);

        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(OptionalJsonRejection::BodyRead)?;

        if bytes.is_empty() {
            return Ok(OptionalJson(None));
        }

        // The body decides, not the header. A non-JSON `Content-Type` is only
        // worth reporting when the bytes really are not JSON for `T`: `curl -d`
        // sends `application/x-www-form-urlencoded` by default with a perfectly
        // good JSON body, and answering 415 for that broke every documented
        // `curl -d` recipe after rc.98 (#2575). When the body is malformed, the
        // header is the more useful thing to name, because a form-encoded body
        // is the common cause.
        match serde_json::from_slice::<T>(&bytes) {
            Ok(value) => Ok(OptionalJson(Some(value))),
            Err(e) => match ct {
                Some(ct_str) if !content_type_is_json(&ct_str) => {
                    Err(OptionalJsonRejection::UnsupportedMediaType(ct_str))
                }
                _ => Err(OptionalJsonRejection::JsonParse(e)),
            },
        }
    }
}

/// `application/json`, `application/json; charset=utf-8`,
/// `application/vnd.elasticsearch+json`, `application/x-ndjson`, `text/plain`
/// and a missing header are all acceptable. Anything else is only reported
/// when the body did not parse.
fn content_type_is_json(ct: &str) -> bool {
    let lower = ct.to_ascii_lowercase();
    lower.contains("json") || lower.starts_with("text/plain")
}

impl<T: Default> OptionalJson<T> {
    /// Convenience for the very common pattern
    /// `body.0.unwrap_or_default()`.
    #[inline]
    pub fn into_or_default(self) -> T {
        self.0.unwrap_or_default()
    }
}

// Deref into the inner `Option<T>` so handlers that previously called
// `body.as_ref()`, `body.iter()`, `body.is_some()`, `body.as_mut()` etc.
// against the stock `Option<Json<T>>` keep working with one mechanical
// change to the type signature only.
impl<T> std::ops::Deref for OptionalJson<T> {
    type Target = Option<T>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for OptionalJson<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;

    async fn extract(
        ct: Option<&str>,
        body: &str,
    ) -> Result<OptionalJson<serde_json::Value>, Response> {
        let mut req = Request::builder().method("POST").uri("/idx/_search");
        if let Some(ct) = ct {
            req = req.header(header::CONTENT_TYPE, ct);
        }
        let req = req.body(Body::from(body.to_owned())).unwrap();
        OptionalJson::<serde_json::Value>::from_request(req, &())
            .await
            .map_err(OptionalJsonRejection::into_response)
    }

    async fn rejection(ct: Option<&str>, body: &str) -> (StatusCode, String) {
        let resp = match extract(ct, body).await {
            Ok(_) => panic!("a {ct:?} body of {body:?} must be rejected"),
            Err(resp) => resp,
        };
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// #2575: `curl -d '{json}'` sends `application/x-www-form-urlencoded` by
    /// default, and rc.98 answered 415 for it — which broke every documented
    /// `curl -d` recipe even though the body was exactly the JSON asked for.
    /// The bytes are what the endpoint reads, so a body that parses is
    /// accepted whatever the header claims.
    #[tokio::test]
    async fn a_json_body_sent_with_curls_default_form_content_type_is_accepted() {
        let body = r#"{"query":{"match_all":{}}}"#;
        let got = extract(Some("application/x-www-form-urlencoded"), body)
            .await
            .expect("a parseable JSON body must be accepted");
        assert_eq!(
            got.0.unwrap()["query"]["match_all"],
            serde_json::json!({}),
            "the body must arrive intact, not defaulted away"
        );
    }

    /// The 415 is still reachable, and when it fires it names the header that
    /// fixes it: the reporter's complaint was a status with no usable reason.
    #[tokio::test]
    async fn a_form_encoded_body_names_the_header_that_would_fix_it() {
        let (status, text) = rejection(Some("application/x-www-form-urlencoded"), "q=1&x=2").await;
        assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert!(
            text.contains("application/x-www-form-urlencoded"),
            "the offending type must be named: {text}"
        );
        assert!(
            text.contains("Content-Type: application/json"),
            "the reason must say which header to send: {text}"
        );
        assert!(
            text.contains("parse_exception") && text.contains("\"status\":415"),
            "the body stays ES-shaped: {text}"
        );
    }

    /// A malformed body with a JSON header is a parse error, not a media-type
    /// error — the header was right, so blaming it would misdirect the reader.
    #[tokio::test]
    async fn malformed_json_under_a_json_content_type_is_a_400() {
        let (status, text) = rejection(Some("application/json"), "{\"query\":").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(text.contains("malformed JSON"), "{text}");
        assert!(!text.contains("Content-Type"), "{text}");
    }

    /// An absent header and an empty body keep their pre-#2575 behaviour: the
    /// first is lenient, the second means "no body".
    #[tokio::test]
    async fn an_absent_header_is_lenient_and_an_empty_body_is_none() {
        assert!(extract(None, r#"{"size":1}"#).await.is_ok());
        assert!(extract(Some("application/json"), "").await.is_ok());
        assert!(extract(Some("application/x-www-form-urlencoded"), "")
            .await
            .is_ok());
    }
}
