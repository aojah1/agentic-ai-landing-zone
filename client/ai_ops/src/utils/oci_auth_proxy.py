
import warnings

warnings.filterwarnings("ignore")
from src.common.config import *  # expects SQLCLI_MCP_PROFILE, FILE_SYSTEM_ACCESS_KEY, TAVILY_MCP_SERVER
import requests
import oci
from oci.signer import Signer

def get_auth():
    PROFILE_NAME = 'DEFAULT'
    config = oci.config.from_file(profile_name=PROFILE_NAME)
    signer = oci.signer.Signer(
        tenancy=config['tenancy'],
        user=config['user'],
        fingerprint=config['fingerprint'],
        private_key_file_location=config['key_file'],
        # passphrase=config.get('passphrase')  # Uncomment if your private key is encrypted
    )
    return signer
def oci_auth_headers(url):
    signer = get_auth()
    request = requests.Request("POST", url, auth=signer, headers={'Content-Type': 'application/json'})
    prepared = request.prepare()
    #print(“Prepared headers before signing:“, prepared.headers)  # Optional: for debugging
    signer(prepared)
    del(prepared.headers['content-length'])
    return prepared.headers

# Get the authenticated headers

url=f"{MCP_SSE_HOST}:{MCP_SSE_PORT}/mcp"
headers = oci_auth_headers(url)
