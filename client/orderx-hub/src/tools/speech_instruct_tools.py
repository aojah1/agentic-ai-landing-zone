import oci

# Connect to any service to integrate any data and functionality with a public REST interface
import requests, os
import json
from oci.addons.adk import Toolkit, tool
from oci.addons.adk import Toolkit, tool
from pathlib import Path
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent
print(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / "config/.env")  # expects OCI_ vars in .env

# Set up the OCI Speech AI endpoint configuration
OCI_COMPARTMENT_ID = os.getenv("OCI_SPEECH_COMPARTMENT_ID")
OCI_SPEECH_BUCKET_NAMESPACE = os.getenv("OCI_SPEECH_BUCKET_NAMESPACE")
OCI_SPEECH_BUCKET= os.getenv("OCI_SPEECH_BUCKET")

def getSigner(profile_name):
    config = oci.config.from_file(profile_name=profile_name)
    signer = oci.signer.Signer(
        tenancy=config["tenancy"],
        user=config["user"],
        fingerprint=config["fingerprint"],
        private_key_file_location=config["key_file"],
        pass_phrase=config.get("pass_phrase")
    )
    return config, signer

def getSpeechClient():
    config, signer = getSigner("DEFAULT") # Change the profile name from DEFAULT, if you are using some other profile
    ai_client = oci.ai_speech.AIServiceSpeechClient(config, signer=signer)
    print(ai_client)
    return ai_client

ai_client = getSpeechClient()

# Give your job related details in these fields
SAMPLE_DISPLAY_NAME = "speech_to_text"
SAMPLE_COMPARTMENT_ID = OCI_COMPARTMENT_ID
SAMPLE_DESCRIPTION = "speech_to_text"
SAMPLE_NAMESPACE = OCI_SPEECH_BUCKET_NAMESPACE
SAMPLE_BUCKET = OCI_SPEECH_BUCKET
JOB_PREFIX = "Python_SDK_DEMO"
FILE_NAMES = ["2025.02.10MLAps-Intro-Anup.mp4",]# "IT Comms OPR Audio -7-16.m4a"]
NEW_COMPARTMENT_ID = OCI_COMPARTMENT_ID
NEW_DISPLAY_NAME = "new_speech_to_text"
NEW_DESCRIPTION = "new_speech_to_text"
# Supported MODEL_TYPE values: ORACLE, WHISPER_MEDIUM
MODEL_TYPE = "WHISPER_MEDIUM"
# Supported language codes for ORACLE MODEL: en-US, en-AU, en-IN, en-GB, it-IT, pt-BR, hi-IN, fr-FR, de-DE, es-ES
LANGUAGE_CODE = "en"
# Supported language codes for WHISPER_MEDIUM MODEL: auto, af, ar, az, be, bg, bs, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gl, he, hi, hr, hu, hy, id, is, it, ja, kk,  kn, ko, lt, lv, # mi, mk, mr, ms, ne, nl, no, pl, pt, ro, ru, sk, sl, sr, sv, sw, ta, th, tl, tr, uk, ur, vi, zh
SAMPLE_MODEL_DETAILS = oci.ai_speech.models.TranscriptionModelDetails(model_type=MODEL_TYPE, domain="GENERIC",  language_code=LANGUAGE_CODE,

transcription_settings=oci.ai_speech.models.TranscriptionSettings(
    diarization=oci.ai_speech.models.Diarization(
        is_diarization_enabled=False    # Boolean value to enable or disable diarization
    )
)
)

SAMPLE_OBJECT_LOCATION = oci.ai_speech.models.ObjectLocation(namespace_name=SAMPLE_NAMESPACE, bucket_name=SAMPLE_BUCKET,
object_names=FILE_NAMES)

SAMPLE_INPUT_LOCATION = oci.ai_speech.models.ObjectListInlineInputLocation(
    location_type="OBJECT_LIST_INLINE_INPUT_LOCATION", object_locations=[SAMPLE_OBJECT_LOCATION])

SAMPLE_OUTPUT_LOCATION = oci.ai_speech.models.OutputLocation(namespace_name=SAMPLE_NAMESPACE, bucket_name=SAMPLE_BUCKET,
                                                             prefix=JOB_PREFIX)

COMPARTMENT_DETAILS = oci.ai_speech.models.ChangeTranscriptionJobCompartmentDetails(compartment_id=NEW_COMPARTMENT_ID)

UPDATE_JOB_DETAILS = oci.ai_speech.models.UpdateTranscriptionJobDetails(display_name=NEW_DISPLAY_NAME, description=NEW_DESCRIPTION)

# Create Transcription Job with details provided
transcription_job_details = oci.ai_speech.models.CreateTranscriptionJobDetails(display_name=SAMPLE_DISPLAY_NAME,
                                                                               compartment_id=SAMPLE_COMPARTMENT_ID,
                                                                               description=SAMPLE_DESCRIPTION,
                                                                               model_details=SAMPLE_MODEL_DETAILS,
                                                                               input_location=SAMPLE_INPUT_LOCATION,
                                                                               output_location=SAMPLE_OUTPUT_LOCATION)

transcription_job = None

@tool
def create_transcription_job()-> str   :
    print("***CREATING TRANSCRIPTION JOB***")
    try:
        transcription_job = ai_client.create_transcription_job(create_transcription_job_details=transcription_job_details)
        print(f"Transcription Job details: {transcription_job.status}")
        list_transcription_tasks(transcription_job.data.id)
    except Exception as e:
        print(e)
    else:
        print(transcription_job.data)

    return transcription_job.data.id

# print("***CANCELLING TRANSCRIPTION JOB***")
# # Cancel transcription job and all tasks under it
# try:
#     ai_client.cancel_transcription_job(transcription_job.data.id)
# except Exception as e:
#     print(e)
#
#
# print("***UPDATING TRANSCRIPTION JOB DETAILS")
# try:
#     ai_client.update_transcription_job(transcription_job.data.id, UPDATE_JOB_DETAILS)
# except Exception as e:
#     print(e)
#
# print("***MOVE TRANSCRIPTION JOB TO NEW COMPARTMENT***")
# try:
#     ai_client.change_transcription_job_compartment(transcription_job.data.id, COMPARTMENT_DETAILS)
# except Exception as e:
#     print(e)
#
#
# print("***GET TRANSCRIPTION JOB WITH ID***")
# # Gets Transcription Job with given Transcription job id
# try:
#     if transcription_job.data:
#         transcription_job = ai_client.get_transcription_job(transcription_job.data.id)
# except Exception as e:
#     print(e)
# else:
#     print(transcription_job.data)
#
#
# print("***GET ALL TRANSCRIPTION JOBS IN COMPARTMENT***")
# # Gets All Transcription Jobs from a particular compartment
# try:
#     transcription_jobs = ai_client.list_transcription_jobs(compartment_id=SAMPLE_COMPARTMENT_ID)
# except Exception as e:
#     print(e)
# else:
#     print(transcription_jobs.data)
#
#
#
@tool
def list_transcription_tasks(id):
    print("***GET ALL TaskS FROM TRANSCIRPTION JOB ID***")
    #Gets Transcription tasks under given transcription Job Id
    transcription_tasks = None
    try:
        transcription_tasks = ai_client.list_transcription_tasks(id)
    except Exception as e:
        print(e)
    else:
        print(transcription_tasks.data)
#
#
# print("***GET PRATICULAR TRANSCRIPTION Task USING JOB AND Task ID***")
# # Gets a Transcription Task with given Transcription task id under Transcription Job id
# transcription_task = None
# try:
#     if transcription_tasks.data:
#
#         transcription_task = ai_client.get_transcription_task(transcription_job.data.id, transcription_tasks.data.items[0].id)
# except Exception as e:
#     print(e)
# else:
#     print(transcription_task.data)
#
#
# print("***CANCEL PARTICULAR TRANSCRIPTION Task***")
# try:
#     if transcription_task:
#         ai_client.cancel_transcription_task(transcription_job.data.id, transcription_task.data.id)
# except Exception as e:
#     print(e)

if __name__ == "__main__":
    #print(transcription_job_details)
    id = create_transcription_job()
