import music21
import time
from music21 import *
from tqdm.notebook import tqdm, trange
import pandas as pd
import logging
import glob
import string
import logging

def preprocessing():
    string = input('Enter the location of the midi folder: ')
    files = glob.glob(string)

    training_notes = []
    training_duration = []

    for file in tqdm(files):
            notes = []
            durations = []

            #parsing MIDI files one by one
            original_score = music21.converter.parse(file).chordify()

            #depending on the element found in the instrument. Like we would have 'rest' for drums,
            #chords for guitar
            for element in original_score.flat:
                note_name = None
                duration_name  = None

                #metadata  
                if isinstance(element, music21.key.Key):
                    note_name = str(element.tonic.name) + ':' + str(element.mode)
                    duration_name = "0.0"
            
                #metadata
                elif isinstance(element, music21.meter.TimeSignature):
                    note_name = str(element.ratioString) + 'TS'
                    duration_name = "0.0"

                elif isinstance(element, music21.chord.Chord):
                    note_name = element.pitches[-1].nameWithOctave
                    duration_name = str(element.duration.quarterLength)

            # As using drums data, elements found would be 'rest'
                elif isinstance(element, music21.note.Rest):
                    note_name = str(element.name)
                    duration_name = str(element.duration.quarterLength)

                elif isinstance(element, music21.note.Note):
                    note_name = str(element.nameWithOctave)
                    duration_name = str(element.duration.quarterLength)

                if note_name and duration_name:
                    notes.append(note_name)
                    durations.append(duration_name)
        
        #notes and duration hold seuence for one music piece. 
                training_notes.append(notes)
                training_duration.append(durations)
    return training_notes,training_duration
        


