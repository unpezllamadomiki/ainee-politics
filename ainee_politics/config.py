"""Application-wide constants and default runtime values."""

DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GKG_URL_TEMPLATE = "http://data.gdeltproject.org/gdeltv2/{bucket}.gkg.csv.zip"

REQUEST_TIMEOUT = 30.0
API_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
GDELT_MIN_INTERVAL_SECONDS = 5.2
GDELT_RATE_LIMIT_MESSAGE = "Please limit requests to one every 5 seconds"
GDELT_SOURCE_LANGUAGE = "english"

USER_AGENT = (
    "Mozilla/5.0 (compatible; TrabajoAineePoliticalCorpusAPI/4.0; +https://gdeltproject.org)"
)