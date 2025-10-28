# retrieve data from application database and perform transactions on application business objects as defined in the application OpenAI Spec
# INVOKE FUSION Agent Studio Agent


import requests
from requests.auth import HTTPBasicAuth
import traceback  # Optional for better debugging
from src.common.config import *


# ---- PARAMETERS ----
API_BASE_URL = AGENT_STUDIO_BASE_URL                                            # <-- Replace with your Environment URL
USERNAME = AGENT_STUDIO_API_USER                                                 # <-- Replace with your username
PASSWORD = AGENT_STUDIO_API_PASS                                                 # <-- Replace with your password

TOKEN_PATH = AGENT_STUDIO_TOKEN_PATH                                              # <-- Path for bearer token
TOKEN_URL = f"{API_BASE_URL}{TOKEN_PATH}"
print(f'TOKEN_URL : {TOKEN_URL}')



AGENT_NAME = "Sales_Order_Assistant_Demo"                                # <-- Replace with your Fusion AI Agent Team Code
AGENT_VERSION = 15                                                              # <-- Replace with version of your published Fusion AI Agent Team
AGENT_PATH = f"/api/fusion-ai/orchestrator/agent/v1/{AGENT_NAME}/invokeAsync"  # <-- Path for agent POST; update if different
AGENT_URL = f"{API_BASE_URL}{AGENT_PATH}"

print(f'AGENT_URL : {AGENT_URL}')

# --------------------

from html.parser import HTMLParser

class HiddenValueExtractor(HTMLParser):
    def __init__(self, target_name="signature"):
        super().__init__()
        self.target = target_name.lower()
        self.value = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "input":
            return
        a = {k.lower(): v for k, v in attrs}
        if a.get("type", "").lower() == "hidden" and a.get("name", "").lower() == self.target:
            self.value = a.get("value")




def get_bearer_token():
    response = requests.get(TOKEN_URL, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    #print("bearer token 1")
    html = response.text
    print(html)
    p = HiddenValueExtractor("signature")
    p.feed(html)
    #print(p.value)  # => the loginCtx value (or None if not found)z


    #print(response.text)
    response.raise_for_status()
    token = p.value
    #token="1If1rzTD15Rawcm//t/uavuP/gTB83t3h6jEWEKcjOUDKminV2L8cLaf6MRkTGXUJ+cMbKj/mGx0X4Vl6+BdMW2/EYg0kaOQISm4J8FuTDOZGtF3AQn8Nqgv3YxLcEe2+i2txl4a1EeAVpTBA1C+Wz3AK73Kzg8YQ2awfYGuswG735whJ1POVO/aSFjOfZG0DRKLiZlL3ABBNafKyk9oO7ko9JoT/kEjIYwjCDSx2zRjAkN6A9PPrWpwCZiyyjFXVXmB13Ej9b55y9APtJKCVBuCuSG1udMPe7jdWdgGkhiZ/Ras0vy+ghjNfUV+nfwh7Q1wO2I6ydw6kMPFhKvaUlDeQoAk4bqtn3q5K1vEt0yS7QpEIWK7rxh85vAzTjjwJpJK96oDpLoyyF4YzQzrgf9xM9Rc8hGxLbaipRZRX7dEr4+IIPIiS0HT9g5UrFn4Bo/cNFqBRYBw2lCXRyyuzw=="
    #token="E1DgvNuHZZlTnTowEoyoIEhGwflit9k3dRhwnhzZqxw0IPPgF5O4AbiT1ZZHMag1M15AJ2loKcbUJJNcG2NufOcjmpVmG8+CK4XXJPFgLTFAZb+Jr7UWwJY2ugPeTbTuz1RurD2p63JX7UHI7/q3FypKeWGvUrSsAyLZd32znirSFg2sHyCBJhe7C6a21E6p5nEGfHru4fGAVlFhPSGFqzzPlsluQI064LcjBYlmX+xQIB+vTlyh3CpheGdSMzm/DHqHroX9pr3PbyV+V1mpzkZBkSWYDKuFx91xTG0dD09LnFct+6zINjHCR9W/6jEvlT6IAx9LhnWjyJdkd2yNMc6oE+7ug7u7yYmq7FXZmM0qbOvMwYjRa/QvGFiR+j1e6UENYbDdVqPXoabOLq/rvDn/xCCSOqWNVNhOpvTLRAB9EbLpWb6uQEFtdrz9knOeXoIKkvKJuA40UoHp4LW0Xg=="
    #print(token)
    if not token:
        raise ValueError("No access_token found in token response")
    return token

def agentCall():
    user_input = "create order for computer service and rentals, Product: Vision Slimline 5100 Tablet, 16 GB, 8\" Display"

    try:
        #bearer_token = get_bearer_token()
        bearer_token = "eyJ4NXQjUzI1NiI6IjF2OVR5UE5TSG5VaGlOWmd0WkQzUEE0SUZfRFl6Tk9JV2VCTGV1OUpCRHMiLCJ4NXQiOiJZcU9pM292Y3ZobVpzM1gzQWREeG1NMHVfZjAiLCJraWQiOiJTSUdOSU5HX0tFWSIsImFsZyI6IlJTMjU2In0.eyJjbGllbnRfb2NpZCI6Im9jaWQxLmRvbWFpbmFwcC5vYzEucGh4LmFtYWFhYWFhYXF0cDViYWF0aTVidzc3bmo0N3d2emEyNGd2Y29tNWZxeXFpc3VtazVxNnVpZmxuamZ0cSIsInN1YiI6InVpRXJyb3IiLCJzaWRsZSI6MzAsInVzZXIudGVuYW50Lm5hbWUiOiJpZGNzLWZiYTU5ZDU0NDQzMzQzOGE4YjliZDBkZDE2MGRlYzRkIiwiaXNzIjoiaHR0cHM6Ly9pZGVudGl0eS5vcmFjbGVjbG91ZC5jb20vIiwiZG9tYWluX2hvbWUiOiJ1cy1waG9lbml4LTEiLCJjYV9vY2lkIjoib2NpZDEudGVuYW5jeS5vYzEuLmFhYWFhYWFhc3VkcHBsZnh1MjN2dG1pc2VoeG5zbmk2ZGY3aDVhaWxsd2JwZ3Vjd3htdnV4bmxuZXRzYSIsImNsaWVudF9pZCI6InVpRXJyb3IiLCJkb21haW5faWQiOiJvY2lkMS5kb21haW4ub2MxLi5hYWFhYWFhYWNham52aTJ4ZmNvNjVmN21henlnbjNuZjdvZ3IyY212Z2syeTUybHF4cWNibGY1eWJiM2EiLCJzdWJfdHlwZSI6ImNsaWVudCIsInNjb3BlIjoidXJuOm9wYzppZG06dC5zZWN1cml0eS5jbGllbnQgdXJuOm9wYzppZG06dC5hcHBzZXJ2aWNlcyB1cm46b3BjOmlkbTp0Lm5hbWVkYXBwYWRtaW4gdXJuOm9wYzppZG06dC51c2VyLmVycm9yIHVybjpvcGM6aWRtOnQudXNlci5hdXRobi5mYWN0b3JzIiwiY2xpZW50X3RlbmFudG5hbWUiOiJpZGNzLW9yYWNsZSIsInJlZ2lvbl9uYW1lIjoidXMtcGhvZW5peC1pZGNzLTQiLCJleHAiOjE3NjAxMDQ2MDIsImlhdCI6MTc2MDEwMTAwMiwiY2xpZW50X2d1aWQiOiIwMjk5ZmZiMzI1ZmU0NGRhYWE1OTdjNDc4MWVhYzc0MyIsImNsaWVudF9uYW1lIjoidWlFcnJvciIsInRlbmFudCI6ImlkY3MtZmJhNTlkNTQ0NDMzNDM4YThiOWJkMGRkMTYwZGVjNGQiLCJqdGkiOiIxMTE5NWUyOWJmMGI0MTc0OWMwZjEzNzAxMWM1ZjJjMiIsImd0cCI6ImNjIiwib3BjIjpmYWxzZSwic3ViX21hcHBpbmdhdHRyIjoidXNlck5hbWUiLCJwcmltVGVuYW50IjpmYWxzZSwidG9rX3R5cGUiOiJBVCIsImF1ZCI6WyJ1cm46b3BjOmxiYWFzOmxvZ2ljYWxndWlkPWlkY3MtZmJhNTlkNTQ0NDMzNDM4YThiOWJkMGRkMTYwZGVjNGQiLCJodHRwczovL2lkY3MtZmJhNTlkNTQ0NDMzNDM4YThiOWJkMGRkMTYwZGVjNGQuaWRlbnRpdHkub3JhY2xlY2xvdWQuY29tIiwiaHR0cHM6Ly9pZGNzLWZiYTU5ZDU0NDQzMzQzOGE4YjliZDBkZDE2MGRlYzRkLnVzLXBob2VuaXgtaWRjcy00LnNlY3VyZS5pZGVudGl0eS5vcmFjbGVjbG91ZC5jb20iXSwiY2FfbmFtZSI6ImRzZmEzMDA0NDY4MiIsImRvbWFpbiI6ImZhLWRhYmxxeS1kZXYxLXFpZWJ0IiwiY2xpZW50QXBwUm9sZXMiOlsiQXV0aGVudGljYXRlZCBDbGllbnQiLCJFcnJvciIsIkNyb3NzIFRlbmFudCJdLCJ0ZW5hbnRfaXNzIjoiaHR0cHM6Ly9pZGNzLWZiYTU5ZDU0NDQzMzQzOGE4YjliZDBkZDE2MGRlYzRkLmlkZW50aXR5Lm9yYWNsZWNsb3VkLmNvbTo0NDMifQ.YSSg0Yyb6h48SJSrOKHtqSxDaeF9X4eNColkBlxmv1Pj7lfTJZ6r8_YZoW3VbLPXM9_B2woWHsb8SloQd6AUdruHYIh7vU2FIslciadG_icDn0Oe42UHGfWGCeEDxJjTFqzGvK_fzXPXgQsNID1ftQxJYs3oNq3QMo3stM5UG-2MLn1YZWSQ2YSYeC9e69zTUhTne6Qz-0hBj2fCCtXx3f_hf-XokQ7Y-KbM7vrCa4cMoUWGlEKXlDRG_mTYUk95mWn6119VYV_rWjE9fbutSoD7nhh6Q9fHnx3adoDKWuCGneVevzpirChaMDaJJBoJGpS1QqOxO-dHbmIVYqShlg"
        #print(bearer_token)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }

        #headers = {"Authorization": f"Bearer {bearer_token}"}
        print(headers)
        payload = {
            "message": user_input,
            "version": AGENT_VERSION
        }

        print("Calling Agent...")
        response = requests.post(AGENT_URL, headers=headers, json=payload, timeout=600)

        print("Got API response, status code:", response.status_code)        
        response.raise_for_status()        
        data = response.json()  # If API response is JSON
        result = data.get("response", "No result found.")
        print("Agent response:", result)
        
    except Exception as e:
        print("API call failed")
         
if __name__ == "__main__":
     agentCall()