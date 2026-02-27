#!/usr/bin/env python3
"""
Axis Mundi — Musical Nervous System
Standalone MIDI composer driven by live city data via OSC.

Receives OSC from axis_mundi_v3.py on port 9001.
Sends MIDI to Ableton Live via virtual MIDI port.

Run alongside v3:
  Terminal 1: python axis_mundi_v3.py
  Terminal 2: python axis_mundi_music.py

Ableton setup:
  - Create virtual MIDI port named "Axis Mundi" (loopMIDI on Windows)
  - Ch 1: Pad/Drone synth (long attack, lush reverb)
  - Ch 2: Arpeggio/Pluck synth (short decay, delay)
  - Ch 3: Solo/Bell synth (crystalline, high register)
  - Ch 4: Sub-bass synth (sine wave, deep)
  - Ch 5: Texture/Atmosphere (granular, noise-based)

Musical philosophy:
  - The city IS the score. We translate its metabolism into harmony.
  - Prediction error = dissonance. When the model is wrong, music becomes tense.
  - When prediction is accurate, music is consonant and flowing.
  - Rain → more reverb, sparser notes. Cold → lower register.
  - Night (few people) → minimal ambient. Day (many) → dense layers.
"""

import time
import math
import random
import threading
import numpy as np

# ─── IMPORTS ───
try:
    import mido
    MIDI_OK = True
except ImportError:
    MIDI_OK = False
    print("[!] pip install mido python-rtmidi")

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    OSC_OK = True
except ImportError:
    OSC_OK = False
    print("[!] pip install python-osc")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Edit these to shape the sound
# ═══════════════════════════════════════════════════════════════

# OSC input (from axis_mundi_v3.py)
OSC_LISTEN_PORT = 9001

# MIDI output
MIDI_PORT_NAME = "Axis"  # Partial match — looks for ports containing this

# Musical parameters
BASE_TEMPO = 72            # BPM — slow, contemplative
ROOT_NOTE = 48             # C3

# Channel assignments (0-indexed)
CH_PAD = 0                 # Lush sustained chords
CH_ARP = 1                 # Rhythmic arpeggios
CH_SOLO = 2                # Sparse melodic notes
CH_BASS = 3                # Sub-bass (vehicles, weight)
CH_TEXTURE = 4             # Atmospheric texture

# Velocity ranges (min, max) — adjust to taste
VEL_PAD = (35, 65)
VEL_ARP = (50, 95)
VEL_SOLO = (60, 90)
VEL_BASS = (80, 115)
VEL_TEXTURE = (25, 50)

# How often each voice plays (in sequencer steps)
# Lower = more frequent. 32 = every 32 steps (~8 bars)
PAD_CHANGE_STEPS = 32      # Chord changes
ARP_DENSITY_RANGE = (2, 8) # Min/max euclidean pulses per 16 steps
SOLO_PROBABILITY = 0.15    # Chance of solo note per eligible step
BASS_INTERVAL = 16         # Steps between bass hits

# Error thresholds for scale changes
ERROR_LOW = 0.15           # Below this = peaceful
ERROR_MED = 0.30           # Above this = tension
ERROR_HIGH = 0.50          # Above this = chaos


# ═══════════════════════════════════════════════════════════════
# CITY STATE (updated by OSC receiver)
# ═══════════════════════════════════════════════════════════════

class CityState:
    """Thread-safe container for live city data."""
    def __init__(self):
        self._lock = threading.Lock()
        self.count = 0
        self.vehicles = 0
        self.speed = 0.0
        self.direction = 0.0
        self.error = 0.0
        self.anomaly = 0
        self.rain = 0.0
        self.temp = 10.0
        self.clusters = 0
        self.roi = [0, 0, 0, 0]
        self.last_update = 0

    def set(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self.last_update = time.time()

    def get(self):
        with self._lock:
            return {
                'count': self.count,
                'vehicles': self.vehicles,
                'speed': self.speed,
                'direction': self.direction,
                'error': self.error,
                'anomaly': self.anomaly,
                'rain': self.rain,
                'temp': self.temp,
                'clusters': self.clusters,
                'roi': list(self.roi),
                'alive': time.time() - self.last_update < 10,
            }

city = CityState()


# ═══════════════════════════════════════════════════════════════
# OSC RECEIVER
# ═══════════════════════════════════════════════════════════════

def setup_osc():
    disp = Dispatcher()
    disp.map("/axis/count", lambda addr, v: city.set(count=int(v)))
    disp.map("/axis/vehicles", lambda addr, v: city.set(vehicles=int(v)))
    disp.map("/axis/speed", lambda addr, v: city.set(speed=float(v)))
    disp.map("/axis/direction", lambda addr, v: city.set(direction=float(v)))
    disp.map("/axis/error", lambda addr, v: city.set(error=float(v)))
    disp.map("/axis/anomaly", lambda addr, v: city.set(anomaly=int(v)))
    disp.map("/axis/rain", lambda addr, v: city.set(rain=float(v)))
    disp.map("/axis/temp", lambda addr, v: city.set(temp=float(v)))
    disp.map("/axis/clusters", lambda addr, v: city.set(clusters=int(v)))
    disp.map("/axis/roi", lambda addr, *v: city.set(roi=list(v)))

    server = ThreadingOSCUDPServer(("0.0.0.0", OSC_LISTEN_PORT), disp)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[OSC] Listening on port {OSC_LISTEN_PORT}")


# ═══════════════════════════════════════════════════════════════
# MUSIC THEORY ENGINE
# ═══════════════════════════════════════════════════════════════

class Harmony:
    """
    Manages scales, chords, and harmonic movement.
    The city's emotional state is expressed through harmony.
    """
    
    SCALES = {
        # Peaceful states
        'major7':    [0, 2, 4, 5, 7, 9, 11],    # Pure, clear, morning
        'lydian':    [0, 2, 4, 6, 7, 9, 11],    # Dreamy, expansive
        
        # Neutral / moving
        'dorian':    [0, 2, 3, 5, 7, 9, 10],    # Cool, flowing, urban
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],   # Warm, groovy
        
        # Tension
        'minor':     [0, 2, 3, 5, 7, 8, 10],    # Melancholy, night
        'phrygian':  [0, 1, 3, 5, 7, 8, 10],    # Dark tension
        
        # Chaos
        'locrian':   [0, 1, 3, 5, 6, 8, 10],    # Unstable, searching
        'diminished': [0, 1, 3, 4, 6, 7, 9, 10], # Maximum dissonance
    }
    
    # Chord progressions for different states
    PROGRESSIONS = {
        'peaceful': [0, 3, 4, 0],         # I - IV - V - I
        'flowing':  [0, 5, 3, 4],         # I - vi - IV - V
        'tension':  [0, 5, 1, 4],         # I - vi - ii - V
        'dark':     [0, 3, 5, 2],         # i - iv - vi - iii
        'chaos':    [0, 6, 2, 5],         # Unpredictable jumps
    }

    def __init__(self):
        self.current_scale = 'dorian'
        self.target_scale = 'dorian'
        self.root = ROOT_NOTE
        self.progression_index = 0
        self.current_progression = 'flowing'
    
    def choose_state(self, error, speed, rain, temp, count):
        """Choose musical state based on city data."""
        # Scale selection
        if error > ERROR_HIGH:
            self.target_scale = 'diminished'
            self.current_progression = 'chaos'
        elif error > ERROR_MED:
            self.target_scale = 'phrygian'
            self.current_progression = 'tension'
        elif rain > 0.5:
            self.target_scale = 'minor'
            self.current_progression = 'dark'
        elif speed > 3.0:
            self.target_scale = 'dorian'
            self.current_progression = 'flowing'
        elif count < 5:
            self.target_scale = 'lydian'
            self.current_progression = 'peaceful'
        else:
            self.target_scale = 'mixolydian'
            self.current_progression = 'flowing'
        
        # Gradual scale transition (don't jump instantly)
        self.current_scale = self.target_scale
        
        # Temperature affects register
        if temp < 5:
            self.root = ROOT_NOTE - 12  # Colder = lower
        elif temp > 20:
            self.root = ROOT_NOTE + 5   # Warmer = brighter
        else:
            self.root = ROOT_NOTE
    
    def note(self, degree):
        """Get MIDI note for a scale degree."""
        scale = self.SCALES[self.current_scale]
        octave = degree // len(scale)
        note_in_scale = scale[degree % len(scale)]
        return max(0, min(127, self.root + octave * 12 + note_in_scale))
    
    def chord(self, root_degree=0):
        """Generate a rich, open voicing chord."""
        prog = self.PROGRESSIONS[self.current_progression]
        base = prog[self.progression_index % len(prog)]
        d = base + root_degree
        
        # Open voicing: root-5th-octave+3rd-octave+7th
        return [
            self.note(d) - 12,        # Deep root
            self.note(d + 2),          # Third
            self.note(d + 4),          # Fifth
            self.note(d + 4) + 12,     # Fifth up an octave
            self.note(d + 6) + 12,     # Seventh (color)
        ]
    
    def advance_chord(self):
        """Move to next chord in progression."""
        self.progression_index += 1
    
    def arp_note(self, index):
        """Get a note for arpeggiation — uses wider range."""
        return self.note(index) + 12  # One octave up from pads
    
    def solo_note(self):
        """Choose a beautiful solo note — pentatonic subset for safety."""
        scale = self.SCALES[self.current_scale]
        # Pick from the most consonant degrees
        safe_degrees = [0, 2, 4, 7, 9]  # Pentatonic-ish
        degree = random.choice(safe_degrees)
        octave_offset = random.choice([24, 36])  # High register
        return max(0, min(127, self.root + scale[degree % len(scale)] + octave_offset))
    
    def bass_note(self):
        """Deep bass note following chord root."""
        prog = self.PROGRESSIONS[self.current_progression]
        base = prog[self.progression_index % len(prog)]
        return max(0, min(127, self.note(base) - 24))  # Two octaves below


# ═══════════════════════════════════════════════════════════════
# RHYTHM ENGINE
# ═══════════════════════════════════════════════════════════════

def euclidean(steps, pulses):
    """Generate euclidean rhythm pattern."""
    if pulses <= 0: return [0] * steps
    if pulses >= steps: return [1] * steps
    return [1 if i * pulses % steps < pulses else 0 for i in range(steps)]


# ═══════════════════════════════════════════════════════════════
# MIDI OUTPUT
# ═══════════════════════════════════════════════════════════════

class MidiOut:
    def __init__(self):
        self.port = None
        if not MIDI_OK:
            print("[MIDI] mido not available")
            return
        
        available = mido.get_output_names()
        print(f"[MIDI] Available ports: {available}")
        
        for name in available:
            if MIDI_PORT_NAME.lower() in name.lower():
                try:
                    self.port = mido.open_output(name)
                    print(f"[MIDI] Connected: {name}")
                    return
                except Exception as e:
                    print(f"[MIDI] Failed to open {name}: {e}")
        
        print(f"[MIDI] No port matching '{MIDI_PORT_NAME}' found!")
        print(f"[MIDI] Create a virtual MIDI port with loopMIDI containing '{MIDI_PORT_NAME}'")
    
    def note_on(self, ch, note, vel):
        if not self.port: return
        note = int(np.clip(note, 0, 127))
        vel = int(np.clip(vel, 0, 127))
        self.port.send(mido.Message('note_on', channel=ch, note=note, velocity=vel))
    
    def note_off(self, ch, note):
        if not self.port: return
        note = int(np.clip(note, 0, 127))
        self.port.send(mido.Message('note_off', channel=ch, note=note, velocity=0))
    
    def cc(self, ch, control, value):
        if not self.port: return
        self.port.send(mido.Message('control_change', channel=ch, 
                                     control=control, value=int(np.clip(value, 0, 127))))
    
    def play(self, ch, note, vel, duration):
        """Non-blocking note with automatic release."""
        self.note_on(ch, note, vel)
        def off():
            time.sleep(duration)
            self.note_off(ch, note)
        threading.Thread(target=off, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
# THE COMPOSER — Main sequencer loop
# ═══════════════════════════════════════════════════════════════

def compose(midi, harmony):
    """
    The urban sequencer. Runs forever, making music from city data.
    Each 'step' is a 16th note at BASE_TEMPO.
    """
    print(f"\n[COMPOSER] Starting at {BASE_TEMPO} BPM")
    print(f"[COMPOSER] Waiting for city data on OSC port {OSC_LISTEN_PORT}...")
    
    step = 0
    step_duration = 60.0 / BASE_TEMPO / 4.0  # 16th note duration
    
    # Smoothed values for gradual transitions
    smooth_error = 0.0
    smooth_speed = 0.0
    smooth_count = 0.0
    prev_anomaly = 0
    
    while True:
        state = city.get()
        
        # Smooth transitions (exponential moving average)
        alpha = 0.08
        smooth_error = smooth_error * (1-alpha) + state['error'] * alpha
        smooth_speed = smooth_speed * (1-alpha) + state['speed'] * alpha
        smooth_count = smooth_count * (1-alpha) + state['count'] * alpha
        
        # Update harmony engine
        harmony.choose_state(
            smooth_error, smooth_speed, 
            state['rain'], state['temp'], smooth_count
        )
        
        # Detect anomaly onset (for dramatic moments)
        anomaly_onset = state['anomaly'] == 1 and prev_anomaly == 0
        prev_anomaly = state['anomaly']
        
        # ─── CONTINUOUS MODULATION (every step) ───
        
        # CC11 Expression: overall intensity follows crowd size
        expression = int(30 + min(smooth_count * 3, 70) + math.sin(step / 12.0) * 10)
        midi.cc(CH_PAD, 11, expression)
        
        # CC74 Filter cutoff: opens with movement speed
        filter_val = int(40 + min(smooth_speed * 12, 60) + smooth_error * 30)
        midi.cc(CH_ARP, 74, filter_val)
        
        # CC91 Reverb: increases with rain
        reverb = int(40 + min(state['rain'] * 40, 60))
        midi.cc(CH_PAD, 91, reverb)
        midi.cc(CH_SOLO, 91, reverb)
        
        # CC1 Mod wheel: tension from prediction error
        mod = int(min(smooth_error * 180, 127))
        midi.cc(CH_PAD, 1, mod)
        midi.cc(CH_TEXTURE, 1, mod)
        
        # CC93 Chorus: cloud cover / weather ambiguity
        # (Estimate from rain — real cloud data available via extended OSC)
        chorus = int(30 + min(state['rain'] * 30, 50))
        midi.cc(CH_PAD, 93, chorus)
        
        # ─── VOICE 1: PADS / DRONES (Ch 1) ───
        # Change chord every N steps (slow harmonic movement)
        if step % PAD_CHANGE_STEPS == 0:
            chord_notes = harmony.chord()
            harmony.advance_chord()
            
            # Velocity depends on time of day / crowd
            vel_base = VEL_PAD[0] + int((VEL_PAD[1]-VEL_PAD[0]) * min(smooth_count/20, 1))
            
            for n in chord_notes:
                # Slight velocity humanization
                vel = vel_base + random.randint(-5, 5)
                duration = step_duration * PAD_CHANGE_STEPS * 0.95  # Hold almost to next chord
                midi.play(CH_PAD, n, vel, duration)
        
        # ─── VOICE 2: ARPEGGIOS (Ch 2) ───
        # Density follows crowd count
        pulses = int(np.clip(
            ARP_DENSITY_RANGE[0] + (smooth_count / 10) * (ARP_DENSITY_RANGE[1]-ARP_DENSITY_RANGE[0]),
            ARP_DENSITY_RANGE[0], ARP_DENSITY_RANGE[1]
        ))
        rhythm = euclidean(16, pulses)
        
        if rhythm[step % 16]:
            # Choose from current scale, influenced by direction
            dir_offset = int(state['direction'] / 60) % 7
            degree = (step % 7 + dir_offset) % 12
            note = harmony.arp_note(degree)
            
            vel = int(np.clip(
                VEL_ARP[0] + smooth_speed * 8 + random.randint(-10, 10),
                VEL_ARP[0], VEL_ARP[1]
            ))
            
            # Duration: shorter when fast, longer when slow
            dur = max(0.1, 0.4 - smooth_speed * 0.03)
            midi.play(CH_ARP, note, vel, dur)
        
        # ─── VOICE 3: SOLO / MELODY (Ch 3) ───
        # Plays sparingly — only when city is relatively calm
        if smooth_error < ERROR_MED and step % 8 == 0:
            if random.random() < SOLO_PROBABILITY:
                note = harmony.solo_note()
                vel = random.randint(*VEL_SOLO)
                dur = random.choice([1.0, 1.5, 2.0, 3.0])  # Long, singing notes
                midi.play(CH_SOLO, note, vel, dur)
        
        # Anomaly onset: dramatic high note
        if anomaly_onset:
            note = harmony.note(7) + 36  # Very high
            midi.play(CH_SOLO, note, 110, 4.0)
            # Also a dissonant cluster
            for offset in [1, 6, 11]:
                midi.play(CH_TEXTURE, harmony.note(offset) + 24, 50, 3.0)
        
        # ─── VOICE 4: BASS (Ch 4) ───
        # Triggered by vehicles and at regular intervals
        if step % BASS_INTERVAL == 0:
            if state['vehicles'] > 0 or smooth_count > 10:
                note = harmony.bass_note()
                vel = int(np.clip(
                    VEL_BASS[0] + state['vehicles'] * 5,
                    VEL_BASS[0], VEL_BASS[1]
                ))
                dur = step_duration * BASS_INTERVAL * 0.7
                midi.play(CH_BASS, note, vel, dur)
        
        # ─── VOICE 5: TEXTURE (Ch 5) ───
        # Sparse atmospheric notes, more active during anomaly
        if step % 4 == 0:
            if smooth_error > ERROR_LOW and random.random() < smooth_error:
                # Dissonant note from the scale edges
                degree = random.choice([1, 3, 5, 6])  # Tension degrees
                note = harmony.note(degree) + random.choice([12, 24])
                vel = random.randint(*VEL_TEXTURE)
                midi.play(CH_TEXTURE, note, vel, random.uniform(0.5, 2.0))
        
        # ─── CLUSTERS on groups ───
        # When people cluster, play gentle chord stabs
        if state['clusters'] > 0 and step % 16 == 8:
            for _ in range(min(state['clusters'], 3)):
                degree = random.randint(0, 6)
                note = harmony.arp_note(degree)
                midi.play(CH_ARP, note, random.randint(30, 50), 0.8)
        
        # ─── Advance ───
        time.sleep(step_duration)
        step += 1
        
        # Periodic log
        if step % 64 == 0 and state['alive']:
            print(f"[♪] {harmony.current_scale} | "
                  f"{state['count']}p {state['vehicles']}v | "
                  f"err:{smooth_error:.0%} | "
                  f"spd:{smooth_speed:.1f} | "
                  f"prog:{harmony.current_progression} | "
                  f"step:{step}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════╗")
    print("║   AXIS MUNDI — Musical Nervous System    ║")
    print("╚══════════════════════════════════════════╝\n")
    
    # Setup OSC receiver
    if OSC_OK:
        setup_osc()
    else:
        print("[!] OSC not available — running with default values")
    
    # Setup MIDI output
    midi = MidiOut()
    
    # Setup harmony engine
    harmony = Harmony()
    
    print(f"\n[CONFIG]")
    print(f"  Tempo: {BASE_TEMPO} BPM")
    print(f"  Root: MIDI note {ROOT_NOTE} ({['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][ROOT_NOTE%12]}{ROOT_NOTE//12-1})")
    print(f"  Channels: Pad={CH_PAD+1} Arp={CH_ARP+1} Solo={CH_SOLO+1} Bass={CH_BASS+1} Texture={CH_TEXTURE+1}")
    print(f"  Error thresholds: peaceful<{ERROR_LOW:.0%} tension>{ERROR_MED:.0%} chaos>{ERROR_HIGH:.0%}")
    print()
    
    try:
        compose(midi, harmony)
    except KeyboardInterrupt:
        print("\n[COMPOSER] Fading out...")
        # Send all-notes-off on all channels
        if midi.port:
            for ch in range(5):
                midi.cc(ch, 123, 0)  # All Notes Off
        print("[COMPOSER] Silence.")


if __name__ == "__main__":
    main()
