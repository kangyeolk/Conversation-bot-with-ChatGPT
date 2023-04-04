import gradio as gr
import openai
from pydub import AudioSegment
from google.cloud import texttospeech
from google.oauth2 import service_account

import os
from dotenv import load_dotenv
load_dotenv()

import uuid

openai.api_key = os.getenv("OPENAI_API_KEY")
credentials = service_account.Credentials.from_service_account_file('my-credentials.json')
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

conversation = []

language_dict = {
    "English": {"en-US", "en-US-Standard-H"},
    "Japanese": {"ja-JP", "ja-JP-Neural2-B"},
    "Korean": {"ko-KR", "ko-KR-Neural2-A"}
}

def clear_conversation():
    conversation = []
    return "conversation has been renewed!"

def transcribe(audio, language, role):
    global conversation
    global tts_client
    
    # Whisper API
    audio_file_wav = open(audio, "rb")
    audio_file_mp3 = AudioSegment.from_wav(audio_file_wav).export("audio.mp3", format="mp3")
    transcript = openai.Audio.transcribe("whisper-1", audio_file_mp3)
    print(transcript)
    
    conversation.insert(0, {"role": "system", "content": f"Role-playing in {language} when you're a {role}"})
    conversation += [{"role": "user", "content": transcript["text"]}]
    
    # Chatgpt API
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=conversation)
    chatgpt_response = response['choices'][0]['message']['content']
    print(chatgpt_response)

    conversation += [{"role": "assistant", "content": chatgpt_response}]

    # Set the language    
    language_code, voice_name = language_dict[language[0]]
    # print(language_code, voice_name)
    
    # Google cloud text-to-speech
    input_text = texttospeech.SynthesisInput(text=chatgpt_response)
    print(input_text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')

    return "output.mp3"


bot = gr.Interface(fn=transcribe, 
            inputs=[
                gr.Audio(source="microphone", type="filepath"),
                gr.CheckboxGroup(["English", "Japanese", "Korean"], label="Language", info="Which language do you practice?"),
                gr.CheckboxGroup(["Teacher", "Clerk", "Friend"], label="Role", info="ChatGPT would be...")], 
            outputs="audio")
bot.launch()