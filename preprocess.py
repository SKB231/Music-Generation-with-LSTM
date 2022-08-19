import os
from tkinter import SINGLE
import music21 as m21
import json


KERN_DATASET_PATH = "deutschl/erk"
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]
SAVE_DIR = "dataset"
MAPPING_PATH = "mapping.json"
SINGLE_FILE_DATASET = "file_dataset.txt"
SEQUENCE_LENGTH = 64

def load_songs_in_kern(dataset_path):
    # go through all the files in the dataset and load them with music 21
    songs = []
    count = 0
    for path, subdir, files in os.walk(dataset_path):
        print(f"About to load {len(files)} files.")
        for file in files:
            if file[-3:] == "krn":
                count+=1
                if count%50==0:
                    print(count)
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs

#We are making sure that the notes in the songs are between the sixteenth note and the "4" note. Present in the above list.
def has_acceptabale_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    
    return True


def encode_song(song, time_step = 0.25):
    # p = 60, d = 1.0 => [60, "_", "_", "_", ]
    # p = 60, d = 0.25 => [60]
    # p = 60, d = 0.5 => [60, "_"]

    # The time_step and d values are in the unit of number (or fraction) of beats.

    encoded_song = []

    for event in song.flat.notesAndRests:
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # midi number like 60
        # handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        
        #convert the notes/rest into time series notation:
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    # cast encoded song to a string:
    encoded_song = " ".join(map(str, encoded_song))

    #map(A, B) => {
    #   for b in B:
    #       return A(B)
    # }
    return encoded_song

def transpose(song):
    ## Music21 Stream contains Different Parts
    ## Parts contain different measures.


    # get key from song:
    
    ##get all parts
    parts = song.getElementsByClass(m21.stream.Part)
    ##Get measures in part 0
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    ##Get first measure, and get note in index 4, which's usually the key
    key = measures_part0[0][4]
    
    # Estimate key using music21:
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")


    # get interval for transposition:
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    # trans[pse song by calculated interval:
    transposed_song = song.transpose(interval)

    return transposed_song

def preprocess(dataset_path):
    # load songs
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i,song in enumerate(songs):
        # filter out songs that have non-acceptable durations
        if not has_acceptabale_durations(song, ACCEPTABLE_DURATIONS):
            continue
        # transpose songs to Cmaj/Amin
        song = transpose(song)
        # encode songs with music time series representation
        encoded_song = encode_song(song)
        if(i%50 == 0):
            print("Encoded count: ", i)
        # save songs to text file
        save_path = os.path.join(SAVE_DIR, str(i))
        save_path += (".txt")
        # with function auto-closes the opened file from save_path
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    
    # remove the last space character from the delimiter
    songs = songs[:-1]

    #save string that contains all datasets
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    
    #save string that contains all datasets
    return songs


def create_mapping(songs, mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split(" ")
    vocabulary = list(set(songs))

    # create mapping:
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
        
    # save the vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    int_songs = []

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    
    # Convert songs to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs

import keras.utils
import numpy as np

def generate_training_seqeunces(sequence_length):
    # load songs and map them to integers
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    
    inputs = []
    targets = []

    # generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one hot encode the sequences
    # input: (# of sequences, sequence length)
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes = vocabulary_size)
    targets = np.array(targets)

    return inputs, targets

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_seqeunces(SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()


