from openai import OpenAI
from openai.types.evals.run_create_params import DataSourceCreateEvalResponsesRunDataSourceSourceFileID

# client = OpenAI(api_key="")
client = OpenAI()

with open("output.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        # model="gpt-4o-transcribe",
        file=audio_file
    )
    print(transcription.text)
