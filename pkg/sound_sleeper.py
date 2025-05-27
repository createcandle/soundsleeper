"""Sound Sleeper adapter for Candle Controller / WebThings Gateway."""

import os
from os import path
import sys
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'lib'))

import json
import time

import threading

from gateway_addon import Adapter, Device, Property, Action, Database



import torch
import numpy as np
import sounddevice as sd
import librosa
from torchvision.models import resnet18
import torch.nn as nn



_TIMEOUT = 3

_CONFIG_PATHS = [
    os.path.join(os.path.expanduser('~'), '.webthings', 'config'),
]

if 'WEBTHINGS_HOME' in os.environ:
    _CONFIG_PATHS.insert(0, os.path.join(os.environ['WEBTHINGS_HOME'], 'config'))


class SoundSleeperAdapter(Adapter):
    """Adapter for Sound Sleeper"""

    def __init__(self, verbose=True):
        """
        Initialize the object.

        verbose -- whether or not to enable verbose logging
        """
        #print("Initialising SoundSleeper")
        self.pairing = False
        self.name = self.__class__.__name__
        self.addon_name = 'soundsleeper'
        Adapter.__init__(self, 'soundsleeper', 'soundsleeper', verbose=verbose)
        #print("Adapter ID = " + self.get_id())

        self.addon_path = os.path.join(self.user_profile['addonsDir'], self.addon_name)

        self.addon_path = os.path.join(self.user_profile['addonsDir'], self.addon_name)

        self.snore_sense_model_path = os.path.join(self.addon_path, 'models','snore_model.pth')

        self.DEBUG = True
        self.running = True
        
        self.interval = 2 # in seconds
        
        self.snoring = False
        
        
        try:
            self.add_from_config()
        except Exception as ex:
            print("Error loading config: " + str(ex))

        
        try:
            sound_sleeper_device = SoundSleeperDevice(self)
            self.handle_device_added(sound_sleeper_device)
            self.devices['sound_sleeper_thing'].connected = True
            self.devices['sound_sleeper_thing'].connected_notify(True)
            self.thing = self.get_device("sound_sleeper_thing")
        except Exception as ex:
            print("Error creating thing: " + str(ex))
            

        
        #while self.running == True:
            #targetProperty = self.thing.find_property('current_description')
            #time.sleep(1)

        #print("creating Snore Sense thread")
        t = threading.Thread(target=self.snore_sense)
        t.daemon = True
        t.start()

        if self.DEBUG:
            print("End of SoundSleeperAdapter init process")
        
        

    def snore_sense(self):
        
        #
        # SnoreSense parts
        #
        
        self.snore_sense_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(f"Using device: {self.snore_sense_device}")

        # Parameters
        sr = 22050
        duration = 3  # 5 seconds
        self.snore_sense_buffer = np.zeros(sr * duration)
        
        
        def load_model(model_path, device):
            model = resnet18(pretrained=False)
            model.fc = nn.Linear(512, 2)  # Adjust for binary classification
            model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model


        # Convert audio to mel spectrogram
        def get_melspectrogram_db(
            audio,
            sr=22050,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmin=20,
            fmax=8300,
            top_db=80,
        ):
            if audio.shape[0] < 5 * sr:
                audio = np.pad(
                    audio, int(np.ceil((5 * sr - audio.shape[0]) / 2)), mode="reflect"
                )
            else:
                audio = audio[: 5 * sr]
            spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
            )
            spec_db = librosa.power_to_db(spec, top_db=top_db)
            return spec_db


        # Normalize the spectrogram
        def spec_to_image(spec, eps=1e-6):
            mean = spec.mean()
            std = spec.std()
            spec_norm = (spec - mean) / (std + eps)
            return spec_norm


        # Make prediction from audio
        def predict(model, audio, device):
            spec = get_melspectrogram_db(audio)
            spec = spec_to_image(spec)
            spec = np.expand_dims(spec, axis=0)  # Add channel dimension
            spec = np.expand_dims(spec, axis=0)  # Add batch dimension
            spec_tensor = torch.tensor(spec, device=device).float()

            with torch.no_grad():
                output = model(spec_tensor)
                _, predicted = torch.max(output, 1)

            return predicted.item()


        


        # Callback function for real-time audio processing
        def audio_callback(indata, frames, time, status):
            #global self.snore_sense_buffer
            if status:
                print(status)

            # Accumulate audio data in the buffer
            self.snore_sense_buffer[:-frames] = self.snore_sense_buffer[frames:]
            self.snore_sense_buffer[-frames:] = indata[:, 0]

            # Check if buffer is full (5 seconds of audio)
            if np.count_nonzero(self.snore_sense_buffer) > 0:
                prediction = predict(model, self.snore_sense_buffer, self.snore_sense_device)
                label = "Snoring" if prediction == 1 else "Non-Snoring"
                #print(f"Prediction: {label}")
                
                if self.snoring != bool(prediction):
                    
                    self.snoring = bool(prediction)
                    self.devices['sound_sleeper_thing'].properties['snoring'].update( self.snoring )
                    #print("snoring changed to: " + str(self.snoring))
                    
        
        
        
        
        
        # Load the trained model
        #snore_sense_model_path = "models/snore_model.pth"
        model = load_model(self.snore_sense_model_path, self.snore_sense_device)
        
        
        with sd.InputStream( callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * 0.5)):
            #print("Listening... ")
            sd.sleep(1000000)  # Keep

        
        
        

    def unload(self):
        print("Shutting down SoundSleeper")
        self.running = False
        


    def remove_thing(self, device_id):
        if self.DEBUG:
            print("-----REMOVING:" + str(device_id))
        
        try:
            obj = self.get_device(device_id)        
            self.handle_device_removed(obj)  # Remove from device dictionary
            if self.DEBUG:
                print("Removed device")
        except Exception as ex:
            if self.DEBUG:
                print("Could not remove thing(s) from devices: " + str(ex))
        
        return



    def add_from_config(self):
        """Attempt to add all configured devices."""
        try:
            database = Database('sound_sleeper')
            if not database.open():
                return

            config = database.load_config()
            database.close()
        except:
            print("Error! Failed to open settings database.")
            return

        if not config:
            print("Error loading config from database")
            return
        
        
        # Debugging
        try:
            if 'Debugging' in config:
                self.DEBUG = bool(config['Debugging'])
                if self.DEBUG:
                    print("Debugging is set to: " + str(self.DEBUG))
            else:
                self.DEBUG = False
        except:
            print("Error loading debugging preference")
            
        
        
        # update frequency
        try:
            if 'Update frequency' in config:
                self.interval = 1 + int(config['Update frequency'])
            else:
                if self.DEBUG:
                    print("Update prequency preference not found")
        except Exception as ex:
            if self.DEBUG:
                print("Update prequency preference not found error: " + str(ex))


#
#  DEVICES
#

class SoundSleeperDevice(Device):
    """SoundSleeper device type."""

    def __init__(self, adapter):
        """
        Initialize the object.
        adapter -- the Adapter managing this device
        """

        
        Device.__init__(self, adapter, 'sound_sleeper_thing')
        #print("Creating SoundSleeper thing")
        
        self._id = 'sound_sleeper_thing'
        self.id = 'sound_sleeper_thing'
        self.adapter = adapter
        self._type.append('BinarySensor')

        self.name = 'sound_sleeper_thing'
        self.title = 'Sleep sounds'
        self.description = 'Sound sleeper thing'


        if self.adapter.DEBUG: 
            print("Empty SoundSleeper thing has been created.")

        self.properties = {}
        # BooleanProperty
        
        self.properties["snoring"] = SoundSleeperProperty(
                        self,
                        "snoring",
                        {
                            "label": "Snoring",
                            'type': 'boolean',
                            'readOnly': True,
                            '@type': 'BooleanProperty',
                        },
                        False)
        
        self.adapter.handle_device_added(self)


        



#
#  PROPERTY
#


class SoundSleeperProperty(Property):
    """SoundSleeper property type."""

    def __init__(self, device, name, description, value):
        
        #print("incoming thing device at property init is: " + str(device))
        Property.__init__(self, device, name, description)
        
        
        self.device = device
        self.name = name
        self.title = name
        self.description = description # dictionary
        self.value = value
        self.set_cached_value(value)
        self.device.notify_property_changed(self)


    def set_value(self, value):
        #print("set_value is called on a SoundSleeper property by the UI. This should not be possible in this case?")
        pass


    def update(self, value):
        
        if value != self.value:
            if self.device.adapter.DEBUG: 
                print("Sound Sleeper property: "  + str(self.title) + ", -> update to: " + str(value))
            self.value = value
            self.set_cached_value(value)
            self.device.notify_property_changed(self)
        else:
            if self.device.adapter.DEBUG: 
                print("Sound Sleeper property: "  + str(self.title) + ", was already this value: " + str(value))

