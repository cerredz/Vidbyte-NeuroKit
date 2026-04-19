from enum import StrEnum


class ProviderBaseUrl(StrEnum):
    DROPBOX_API = "https://api.dropboxapi.com/2"
    DROPBOX_CONTENT = "https://content.dropboxapi.com/2"
    GOOGLE_DRIVE_API = "https://www.googleapis.com/drive/v3"
    META_GRAPH_API = "https://graph.facebook.com"
    SLACK_API = "https://slack.com/api"
    VIMEO_API = "https://api.vimeo.com"
