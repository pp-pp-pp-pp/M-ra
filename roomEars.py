import os
import json
from openai import OpenAI

# Set your OpenAI API key
api_key = ''

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Define the directory to save conversations
conversation_dir = './llark-gpt2-conversation'
os.makedirs(conversation_dir, exist_ok=True)

# Define the prompt template
prompt_template = """
You are an expert AI assistant that is knowledgeable about music production, musical structure, music history, and music styles, and you are hearing audio of a short clip or loop extracted from a piece of music. What you hear is described in the .txt-formatted outputs below, describing the same audio clip you are listening to. Answer all questions as if you are hearing the audio clip. This description is provided in a .txt dictionary, where the keys and values represent attributes of the music clip.

The .txt also contains the following annotations:
    - defined_tempo: indicator for whether the clip has a well-defined tempo
    - suade: music macrogenre 'suade' associated with this clip. Options are Classical, Theatrical, Traditional, World, Rock, Hip Hop, EDM, Experimental
    - description: an optional description of the "sound of" the clip
    - tags: an optional set of tags associated with the clip
    - name: the original name of the clip
    - origin: the name or the "origin" of the clip 
    - instrumentation_percussion: indicator for whether the clip contains percussion
    - instrumentation_bass: indicator for whether the clip contains bass
    - instrumentation_chords: indicator for whether the clip contains chords
    - instrumentation_melody: indicator for whether the clip contains a melody
    - instrumentation_fx: indicator for whether the clip is "fx" (or "sound effects")
    - instrumentation_vocal: indicator for whether the clip contains vocals
    - time_signature: the time signature of the clip
    - tempo_in_beats_per_minute_madmom: the tempo of the track in beats per minute (BPM).
    - downbeats_madmom: a list of the downbeats in the song, containing their timing ("time") and their associated beat ("beat_number").
      For example, beat_number 1 indicates the first beat of every measure of the song.
      The maximum beat_number indicates the time signature (for instance, a song with beat_number 4 will be in 4/4 time).
    - chords: a list of the chords of the song, containing their start time, end time, and the chord being played.
    - chords roman: a list of chords of the song, in roman numeral form. 
    - key: the key of the song.
    - Mida and DEF Notation Stems: notation for the songs
    - vocal register: the vocal range of the singer or lead melody
    - ai flag: whether or not the song was written by a human, or an ai, you will be provided this
    - lyrics: the lyrics which will be provided to you, you can quote a memorable line or two, but dont tell the whole lyrics
    - lyrics reaction: a summary of the lyrical themes, not provided to you. make up with your knowledge of them. if the lyrics have not been provided, you dont need to answer anything here.

Ignore any other fields besides the ones described above.

Provide a detailed musical description of the clip, from the perspective of a musical expert describing the clip as they hear it being played.
Make sure to describe the musical style, any unique features of the clip, its chords and tempo, the instruments used (if this information is in the metadata), etc.

The answers should be in a tone that an AI assistant is hearing the music and describing it to a listener who wants a brief summary to understand everything in the clip.

Only provide details that are based on the provided metadata or your background knowledge of music as an intelligent AI assistant.
Explain any musical concepts that would be unfamiliar to a non-musician.
Do not specifically reference the provided metadata in the response; instead, respond as if you are hearing the song and reporting a rich description of what you hear.
The descriptions should keep in mind that this may only be a short clip, loop, or part of a song, and not the complete song.

IMPORTANT!! Do not use the word "metadata" anywhere in the answers to the questions. DO NOT disclose that metadata about the song is provided to you.
DO NOT mention the name of the clip, or the pack_name. Do not reveal that you know details of how the song was produced; instead, use phrases like "it sounds like XXX instrument" or "what I hear might be a YYY microphone".

---

Below is the .txt description of the audio clip:

{metadata}
"""

# Read metadata inputs from a file (e.g., 'metadata_inputs.txt')
# Each metadata input should be separated by a delimiter (e.g., '---')
metadata_file = 'metadata_inputs.txt'

# Function to read metadata inputs from the file
def read_metadata_inputs(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    # Split the content by delimiter
    metadata_list = content.strip().split('---')
    return [md.strip() for md in metadata_list if md.strip()]

# Function to generate response using OpenAI API (new version)
def generate_response(prompt):
    completion = client.completions.create(model='gpt-3.5-turbo-instruct', prompt=prompt)
    return completion.choices[0].text.strip()

# Main function
def main():
    metadata_inputs = read_metadata_inputs(metadata_file)
    
    for idx, metadata in enumerate(metadata_inputs):
        # Format the prompt with the current metadata
        prompt = prompt_template.format(metadata=metadata)
        
        # Generate the response from GPT
        response = generate_response(prompt)
        
        # Prepare the conversation text
        conversation_text = f"USER:\n{metadata}\n\nGPT:\n{response}\n"
        
        # Save the conversation to a text file
        conversation_file = os.path.join(conversation_dir, f'conversation_{idx+1}.txt')
        with open(conversation_file, 'w') as f:
            f.write(conversation_text)
        
        print(f"Saved conversation {idx+1} to {conversation_file}")

if __name__ == '__main__':
    main()
