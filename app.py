from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import RTCConfiguration

RTC_CONFIGURATION = RTCConfiguration({
      "RTCIceServer": [{
        "urls": ["turn:turn.chosangnimiswatching.ml:3478"],
        "username": "guest",
        "credential": "somepassword",
      }]}
)

webrtc_streamer(key="example", rtc_configuration=RTC_CONFIGURATION)