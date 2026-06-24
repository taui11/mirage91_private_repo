#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
 Motor-Imagery cue paradigm for EEG  --  PsychoPy (Coder / standalone .py)
============================================================================

TWO DISPLAY MODES (selected at startup):

  CROSS mode — timing driven by cross overlaps
    The second cross rotates continuously. Every time it aligns with the
    fixed cross (every 90 deg) counts as one "overlap tick".
    overlap period = tick_time seconds.

    Trial timeline:
      [0 .. t_fadein)               fade IN    (precue trigger at start)
      [t_fadein .. t_fadein+t_stay) stay       (MI trigger at t_fadein)
      [t_fadein+t_stay .. total)    fade OUT
      n_pause ticks of blank cross  (pause trigger at fade-out end)
      n_cooldown ticks before first trial so participant can settle.

  CIRCLE mode — continuous breathing circle
    A circle oscillates r_max → 0 → r_max using a cosine curve.
    Period = (t_fadein + t_stay + t_fadeout + n_pause) * tick_time seconds.
    Images appear as the circle shrinks below r_precue, fully visible
    inside r_ring. The circle never stops — n_pause is the "rest" portion
    at the top of each breath. n_cooldown seconds of animated circle
    play before the first trial.

  BIGPICTURE option (available for both modes):
    Image is centred and large instead of small cardinal positions.

HOW TO RUN
  1. Install PsychoPy  (standalone app, or  pip install psychopy).
  2. Put 4 images in an "images/" folder next to this file.
     Missing images -> text placeholder so you can test timing.
  3. PsychoPy Coder -> Open -> Run,  or  python mi_eeg_paradigm.py
  4. ESC to abort (markers saved).

EEG MARKERS  (trigger_backend: 'lsl' | 'parallel' | 'none')
  Value  Label              When fired
  -----  -----------------  ------------------------------------------------
    250  run_start          First flip of each run (before cooldown)
    251  run_end            First flip after last trial of each run

     11  precue_rest  ]     At trial start — before image appears.
     12  precue_left  ]     Epoch on these for condition-locked analysis.
     13  precue_right ]
     14  precue_feet  ]

     21  mi_rest  ]         CROSS: when elapsed >= t_fadein (start of stay).
     22  mi_left  ]         CIRCLE: first frame radius crosses r_ring inward.
     23  mi_right ]         Use these as the MI onset marker for EEG epochs.
     24  mi_feet  ]

     30  pause              CROSS: immediately after fade-out, before blank ticks.
                            CIRCLE: first frame radius crosses r_precue outward.
                            Also sent once at cooldown start (before first trial).

  All markers are also written to a CSV log regardless of trigger_backend.
============================================================================
"""

from psychopy import visual, core, event, gui, monitors, logging
import os
import csv
import random
import datetime

logging.console.setLevel(logging.WARNING)


# ===========================================================================
#  PARAMETERS
# ===========================================================================
P = {
    # --- session ---
    'participant'     : 'sub-01',
    'session'         : '001',
    'n_runs'          : 3,
    'n_repeats'       : 10,        # per class per run -> 4*10 = 40 trials/run

    # --- timing (all in overlap ticks; 1 tick = tick_time seconds) ---
    # Trial: fade-in -> stay -> fade-out -> pause (cross only)
    't_fadein'        : 0.5,       # ticks for picture to fade in
    't_stay'          : 1.0,       # ticks picture stays fully visible
    't_fadeout'       : 0.5,       # ticks for picture to fade out
    'n_pause'         : 1,         # whole ticks of pause (cross only) after fade-out
    'n_cooldown'      : 1,         # ticks of just-cross before first trial
    'fadein_gamma'    : 7.5,       # fade-in curve:  >1=slow start, 1=linear, <1=fast start
    'fadeout_gamma'   : 0.75,       # fade-out curve: <1=stays bright then drops, 1=linear

    # --- cross appearance ---
    'cross_size'      : 0.25,      # half-length of each arm  (height units)
    'cross_thickness' : 0.003,     # half-width of each bar   (height units)

    # --- rotating cross ---
    'tick_time'       : 3.0,       # seconds per overlap tick  (rot_speed = 90/tick_time deg/s)
    'overlap_tol'     : 3.0,       # degrees for overlap detection

    # --- picture appearance ---
    'bigpicture'      : False,     # True  = faint overlay;  False = small cardinal positions
    'overlay_opacity' : 0.40,      # peak opacity when bigpicture = True
    'small_opacity'   : 0.7,       # peak opacity when bigpicture = False
    'small_size'      : (0.225, 0.225),
    # big_size and small_size are per-class; see BIG_SIZE / SMALL_SIZE below

    # --- circle mode (third display mode: small cardinal pics + pulsing circle) ---
    # Continuous cosine oscillation: r = r_max·cos²(π·t/T), T = TICKS_PER_TRIAL·tick_time.
    # Static red ring at r_ring is always visible and marks the MI threshold.
    # Images appear as precue when circle radius < r_precue, fully visible when < r_ring.
    'circle_mode'     : False,   # True = circle/ring display replaces rotating cross
    'circle_r_max'    : 0.20,    # max radius of moving circle (height units)
    'circle_r_ring'   : 0.07,    # fixed ring radius (< r_max); marks MI onset
    'circle_r_precue' : 0.13,    # image starts fading in when circle shrinks below this
    'circle_line_w'   : 4,       # ring line width (pixels)
    'circle_opacity'  : 0.5,     # peak image opacity in circle mode

    # --- hardware / display ---
    'fullscreen'      : True,
    'screen'          : 0,
    'win_size'        : (1920, 1080),
    'bg_color'        : 'black',
    'fg_color'        : 'white',
    'use_photodiode'  : False,
    'trigger_backend' : 'lsl',    # 'none' | 'lsl' | 'parallel'
    'parallel_address': 0x0378,

    # --- misc ---
    'rng_seed'        : None,
    'images_dir'      : 'images',
    'avoid_repeats'   : True,
    'lead_in'         : 0.5,
}

IMAGE_FILES = {
    'rest' : 'rest_picture.png',
    'left' : 'left_hand_pic.png',
    'right': 'right_hand_pic.png',
    'feet' : 'feet_pic.png',
}
CLASSES  = ['rest', 'left', 'right', 'feet']

# Cardinal positions for small-picture mode
# rest=top, left=left, right=right, feet=bottom
SMALL_POS = {
    'rest'  : ( 0.00,  0.40),
    'left'  : (-0.45,  0.00),
    'right' : ( 0.45,  0.00),
    'feet'  : ( 0.00, -0.40),
}

# Per-class display sizes (height units). left/right are 2:1 landscape.
BIG_SIZE = {
    'rest'  : (0.45, 0.45),
    'left'  : (0.90, 0.45),
    'right' : (0.90, 0.45),
    'feet'  : (0.45, 0.45),
}

SMALL_SIZE = {
    'rest'  : (0.225, 0.225),
    'left'  : (0.450, 0.225),
    'right' : (0.450, 0.225),
    'feet'  : (0.225, 0.225),
}

TRIG = {
    'run_start': 250,
    'run_end'  : 251,
    'pause'    : 30,
    'precue'   : {'rest': 11, 'left': 12, 'right': 13, 'feet': 14},
    'mi'       : {'rest': 21, 'left': 22, 'right': 23, 'feet': 24},
}


# ===========================================================================
#  STARTUP DIALOG  (two steps)
# ===========================================================================

# --- Step 1: display mode ---
_dlg1_info = {
    'cue_style'  : ['circle', 'cross'],
    'bigpicture' : False,
}
_dlg1 = gui.DlgFromDict(_dlg1_info, title='MI EEG – Display Mode',
                         order=['cue_style', 'bigpicture'])
if not _dlg1.OK:
    core.quit()
P['circle_mode'] = (_dlg1_info['cue_style'] == 'circle')
if isinstance(_dlg1_info['bigpicture'], str):
    P['bigpicture'] = _dlg1_info['bigpicture'] in ('True', 'true', '1')
else:
    P['bigpicture'] = bool(_dlg1_info['bigpicture'])
_mode_label = _dlg1_info['cue_style'] + (' + bigpicture' if P['bigpicture'] else '')

# --- mode-specific defaults (overwrite P before dialog 2 reads it) ---
_DEFAULTS_CROSS = {
    't_fadein'     : 0.5,
    't_stay'       : 1.0,
    't_fadeout'    : 0.5,
    'n_pause'      : 1,
    'n_cooldown'   : 1,
    'fadein_gamma' : 7.5,
    'fadeout_gamma': 0.75,
    'tick_time'    : 3.0,
}
_DEFAULTS_CIRCLE = {
    't_fadein'     : 0.5,
    't_stay'       : 1.0,
    't_fadeout'    : 0.5,
    'n_pause'      : 1,
    'n_cooldown'   : 1,
    'fadein_gamma' : 2.0,
    'fadeout_gamma': 0.5,
    'tick_time'    : 3.0,
}
P.update(_DEFAULTS_CIRCLE if P['circle_mode'] else _DEFAULTS_CROSS)

# --- Step 2: session & timing parameters ---
_dlg2_info = {
    'participant'    : P['participant'],
    'session'        : P['session'],
    'n_runs'         : P['n_runs'],
    'n_repeats'      : P['n_repeats'],
    't_fadein'       : P['t_fadein'],
    't_stay'         : P['t_stay'],
    't_fadeout'      : P['t_fadeout'],
    'fadein_gamma'   : P['fadein_gamma'],
    'fadeout_gamma'  : P['fadeout_gamma'],
    'n_pause'        : P['n_pause'],
    'n_cooldown'     : P['n_cooldown'],
    'tick_time'      : P['tick_time'],
    'avoid_repeats'  : P['avoid_repeats'],
    'fullscreen'     : P['fullscreen'],
    'trigger_backend': ['lsl', 'none', 'parallel'],
}
_order2 = ['participant', 'session', 'n_runs', 'n_repeats',
           't_fadein', 't_stay', 't_fadeout', 'fadein_gamma', 'fadeout_gamma',
           'n_pause', 'n_cooldown', 'tick_time',
           'avoid_repeats', 'fullscreen', 'trigger_backend']
# circle mode: n_cooldown is in seconds not ticks (label stays the same)
# cross mode: n_cooldown is tick-based — no difference in dialog, just semantics
_dlg2 = gui.DlgFromDict(_dlg2_info, title='MI EEG – Settings  [mode: %s]' % _mode_label,
                          order=_order2)
if not _dlg2.OK:
    core.quit()
P.update(_dlg2_info)

# --- force correct types (DlgFromDict returns strings) ---
for k in ('n_runs', 'n_repeats', 'n_pause', 'n_cooldown'):
    P[k] = int(P[k])
for k in ('tick_time', 't_fadein', 't_stay', 't_fadeout', 'fadein_gamma', 'fadeout_gamma',
          'overlap_tol', 'cross_size', 'cross_thickness',
          'overlay_opacity', 'lead_in'):
    P[k] = float(P[k])
if isinstance(P['avoid_repeats'], str):
    P['avoid_repeats'] = P['avoid_repeats'] in ('True', 'true', '1')
if isinstance(P['fullscreen'], str):
    P['fullscreen'] = P['fullscreen'] in ('True', 'true', '1')

overlap_period = float(P['tick_time'])          # seconds per overlap tick
P['rot_speed']  = 90.0 / overlap_period         # deg/s used internally
TICKS_PER_TRIAL = P['t_fadein'] + P['t_stay'] + P['t_fadeout'] + P['n_pause']
print('[INFO] Overlap period = %.3f s' % overlap_period)
print('[INFO] Trial duration = %d overlaps = %.1f s'
      % (TICKS_PER_TRIAL, TICKS_PER_TRIAL * overlap_period))
print('[INFO] Cooldown = %d overlaps = %.1f s'
      % (P['n_cooldown'], P['n_cooldown'] * overlap_period))
print('[INFO] Trials per run = %d,  runs = %d,  total = %d'
      % (4 * P['n_repeats'], P['n_runs'], 4 * P['n_repeats'] * P['n_runs']))


# ===========================================================================
#  HELPERS
# ===========================================================================
class QuitExperiment(Exception):
    pass


class TriggerBox:
    def __init__(self, backend, address=0x0378):
        self.backend = backend
        self.outlet = None
        self.port = None
        self.pending = False
        if backend == 'lsl':
            try:
                from pylsl import StreamInfo, StreamOutlet
                info = StreamInfo('PsychoPyMarkers', 'Markers', 1, 0,
                                  'int32', 'mi_paradigm_01')
                self.outlet = StreamOutlet(info)
            except Exception as e:
                print('[WARN] LSL unavailable (%s) -> log-only.' % e)
                self.backend = 'none'
        elif backend == 'parallel':
            try:
                from psychopy import parallel
                self.port = parallel.ParallelPort(address=address)
                self.port.setData(0)
            except Exception as e:
                print('[WARN] Parallel port unavailable (%s) -> log-only.' % e)
                self.backend = 'none'

    def send(self, code, label=''):
        t = globalClock.getTime()
        MARKERS.append([P['participant'], P['session'], CUR['run'], CUR['trial'],
                        round(t, 5), int(code), label])
        if self.backend == 'lsl' and self.outlet is not None:
            self.outlet.push_sample([int(code)])
        elif self.backend == 'parallel' and self.port is not None:
            self.port.setData(int(code))
            self.pending = True

    def clear(self):
        if self.backend == 'parallel' and self.port is not None and self.pending:
            self.port.setData(0)
            self.pending = False


def flip():
    trig.clear()
    win.flip()


def check_quit():
    if event.getKeys(keyList=['escape']):
        raise QuitExperiment()


def make_cross(color='white'):
    L = P['cross_size']
    w = P['cross_thickness']
    verts = [( w,  w), ( w,  L), (-w,  L), (-w,  w),
             (-L,  w), (-L, -w), (-w, -w), (-w, -L),
             ( w, -L), ( w, -w), ( L, -w), ( L,  w)]
    return visual.ShapeStim(win, vertices=verts, closeShape=True,
                            fillColor=color, lineColor=color, units='height')


# --- continuous rotation ---------------------------------------------------
def advance_rotation():
    dt = frameClock.getTime()
    frameClock.reset()
    rot_cross.ori = (rot_cross.ori + P['rot_speed'] * dt) % 360.0


def _is_overlapping():
    tol = P['overlap_tol']
    ori = rot_cross.ori % 90.0
    return ori < tol or ori > (90.0 - tol)


def _draw_circle_scene(r=None):
    """Draw the breathing circle at radius r (defaults to r_max = idle state)."""
    if r is None:
        r = P['circle_r_max']
    if r > 0.002:
        breath_circle.radius = r
        breath_circle.draw()
    mi_ring.draw()


def draw_scene(extra=None, photodiode_on=False, circle_r=None):
    advance_rotation()
    if P['circle_mode']:
        _draw_circle_scene(r=circle_r)
    else:
        fixation.draw()
        rot_cross.draw()
    if extra is not None:
        extra.draw()
    if P['use_photodiode'] and photodiode_on:
        pd.draw()


def wait_for_overlap():
    """Spin until the next rising-edge overlap (was outside -> enters zone).
    Returns immediately if we're already NOT overlapping and then enter."""
    # first make sure we leave any current overlap zone
    while _is_overlapping():
        draw_scene()
        flip()
        check_quit()
    # now wait to enter
    while not _is_overlapping():
        draw_scene()
        flip()
        check_quit()


def show_text(msg, keys=('space',)):
    txt.text = msg
    event.clearEvents()
    while True:
        txt.draw()
        flip()
        k = event.getKeys(keyList=list(keys) + ['escape'])
        if 'escape' in k:
            raise QuitExperiment()
        if any(x in k for x in keys):
            break
    event.clearEvents()


def save_markers():
    if not MARKERS:
        return
    os.makedirs('data', exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fn = os.path.join('data', '%s_%s_%s_markers.csv'
                      % (P['participant'], P['session'], stamp))
    with open(fn, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['participant', 'session', 'run', 'trial',
                    't_sec', 'code', 'label'])
        w.writerows(MARKERS)
    print('[INFO] Markers saved to %s' % fn)


# ===========================================================================
#  TRIAL PHASES  (all overlap-counted)
# ===========================================================================

def _wait_n_ticks(n):
    """Wait for n overlap ticks, tracking the last one in _tick_start."""
    for _ in range(n):
        _wait_for_overlap_tracked()


def phase_cooldown():
    """Wait n_cooldown ticks (cross) or seconds (circle) before first trial."""
    win.callOnFlip(trig.send, TRIG['pause'], 'cooldown')
    if P['circle_mode']:
        import math as _math
        t_wait = P['n_cooldown'] * overlap_period
        _clk = core.Clock()
        while _clk.getTime() < t_wait:
            elapsed = _clk.getTime()
            phase = (elapsed / overlap_period) % 1.0
            r = P['circle_r_max'] * 0.5 * (1.0 + _math.cos(2.0 * _math.pi * phase))
            draw_scene(circle_r=r)
            flip()
            check_quit()
    else:
        _wait_n_ticks(P['n_cooldown'])


def _tick_time_frac():
    """
    Return a float in [0, 1] representing how far we are through the current
    overlap tick, estimated from elapsed wall-clock time since last overlap.
    Used for smooth sub-tick opacity interpolation.
    """
    # We track tick start time via a module-level variable updated each overlap.
    elapsed = globalClock.getTime() - _tick_start[0]
    return min(elapsed / overlap_period, 1.0) if overlap_period > 0 else 1.0


# Module-level mutables (lists avoid global statements)
_tick_start  = [0.0]


def _wait_for_overlap_tracked():
    """Like wait_for_overlap() but records the tick-start time."""
    while _is_overlapping():
        draw_scene()
        flip()
        check_quit()
    while not _is_overlapping():
        draw_scene()
        flip()
        check_quit()
    _tick_start[0] = globalClock.getTime()


def run_trial(cls):
    """
    One full trial for class `cls`.

    Picture timeline (all durations in ticks = multiples of tick_time seconds):
      [0 .. t_fadein)              fade IN   (precue trigger at start)
      [t_fadein .. t_fadein+t_stay) stay     (MI trigger at t_fadein)
      [t_fadein+t_stay .. total)   fade OUT
    Then waits for the next overlap and counts n_pause ticks (cross only).
    """
    if P['circle_mode']:
        run_trial_circle(cls)
        return

    peak      = P['overlay_opacity'] if P['bigpicture'] else P['small_opacity']
    t_fi      = P['t_fadein']  * overlap_period   # seconds
    t_st      = P['t_stay']    * overlap_period
    t_fo      = P['t_fadeout'] * overlap_period
    t_total   = t_fi + t_st + t_fo

    stim = _get_stim(cls)
    stim.opacity = 0.0
    mi_fired = False

    trig.send(TRIG['precue'][cls], 'precue_%s' % cls)
    t0 = _tick_start[0]

    # ---- Picture phase: runs for t_total seconds, frame by frame -------------
    while True:
        elapsed = globalClock.getTime() - t0

        if elapsed >= t_fi and not mi_fired:
            trig.send(TRIG['mi'][cls], 'mi_%s' % cls)
            mi_fired = True

        if elapsed < t_fi:
            t = (elapsed / t_fi) if t_fi > 0 else 1.0
            stim.opacity = peak * (t ** P['fadein_gamma'])
        elif elapsed < t_fi + t_st:
            stim.opacity = peak
        elif elapsed < t_total:
            t = (t_total - elapsed) / t_fo if t_fo > 0 else 0.0
            stim.opacity = peak * (t ** P['fadeout_gamma'])
        else:
            stim.opacity = 0.0

        show_pic = stim.opacity > 0.01
        draw_scene(extra=stim if show_pic else None, photodiode_on=show_pic)
        flip()
        check_quit()

        if elapsed >= t_total:
            break

    stim.opacity = 0.0

    # ---- Pause ticks: cross only ---------------------------------------------
    trig.send(TRIG['pause'], 'pause')
    _wait_n_ticks(P['n_pause'])


def run_trial_circle(cls):
    """
    The circle breathes continuously: r_max → 0 → r_max in one period.
    Period = (t_fadein + t_stay + t_fadeout + n_pause) * overlap_period seconds.
    Sine curve keeps the motion smooth with no holds or jumps.

    LSL markers fire at radius thresholds:
      precue  — at trial start (circle begins shrinking)
      mi      — first frame circle radius crosses r_ring inward
      pause   — first frame circle radius crosses r_precue outward (growing back)

    Image opacity ramps 0→peak as r goes r_precue→r_ring, then stays peak,
    then ramps peak→0 as r goes r_ring→r_precue on the way back out.
    """
    import math as _math

    peak  = P['overlay_opacity'] if P['bigpicture'] else P['circle_opacity']
    r_max = P['circle_r_max']
    r_pc  = P['circle_r_precue']
    r_rg  = P['circle_r_ring']

    period = (P['t_fadein'] + P['t_stay'] + P['t_fadeout'] + P['n_pause']) * overlap_period

    stim = _get_stim(cls)
    stim.opacity = 0.0
    mi_sent    = False
    pause_sent = False

    trig.send(TRIG['precue'][cls], 'precue_%s' % cls)
    trial_clock = core.Clock()

    while True:
        elapsed = trial_clock.getTime()
        # sine goes 1 → -1 → 1 over one period; map to r_max → 0 → r_max
        phase = elapsed / period          # 0→1 over one breath
        r = r_max * 0.5 * (1.0 + _math.cos(2.0 * _math.pi * phase))
        shrinking = phase < 0.5

        # --- triggers at radius thresholds ---
        if shrinking and r <= r_rg and not mi_sent:
            trig.send(TRIG['mi'][cls], 'mi_%s' % cls)
            mi_sent = True
        if not shrinking and r >= r_pc and not pause_sent:
            trig.send(TRIG['pause'], 'pause')
            pause_sent = True

        # --- image opacity driven by radius ---
        if r <= r_rg:
            stim.opacity = peak
        elif r < r_pc:
            t = (r_pc - r) / (r_pc - r_rg)   # 0 at r_pc, 1 at r_rg
            gamma = P['fadein_gamma'] if shrinking else P['fadeout_gamma']
            stim.opacity = peak * (t ** gamma)
        else:
            stim.opacity = 0.0

        show_pic = stim.opacity > 0.01
        draw_scene(extra=stim if show_pic else None,
                   photodiode_on=show_pic, circle_r=r)
        flip()
        check_quit()

        if elapsed >= period:
            break

    stim.opacity = 0.0


def _get_stim(cls):
    """Return the image stim (or text placeholder) for a class, configured."""
    is_big = P['bigpicture']
    pos = (0.0, 0.0) if is_big else SMALL_POS[cls]
    if IMG_STIMS[cls] is not None:
        stim = IMG_STIMS[cls]
        stim.size = BIG_SIZE[cls] if is_big else SMALL_SIZE[cls]
        stim.pos  = pos
    else:
        stim = placeholder
        stim.text   = cls.upper()
        stim.height = 0.15 if is_big else 0.09
        stim.pos    = pos
    return stim


# --- trial sequence --------------------------------------------------------
def _shuffle_no_repeats(seq, forbidden_first=None):
    """
    Return a shuffled copy of seq with no two adjacent identical elements.
    forbidden_first: if set, seq[0] must not equal this value (run boundary).
    Uses a greedy approach: always pick a random element that is not the same
    as the previous one, backtracking if stuck.
    """
    counts = {}
    for x in seq:
        counts[x] = counts.get(x, 0) + 1

    result = []
    prev = forbidden_first

    def build(counts, prev):
        if not any(counts.values()):
            return True
        candidates = [c for c, n in counts.items() if n > 0 and c != prev]
        if not candidates:
            return False
        random.shuffle(candidates)
        for c in candidates:
            result.append(c)
            counts[c] -= 1
            if build(counts, c):
                return True
            result.pop()
            counts[c] += 1
        return False

    if build(counts, prev):
        return result
    # fallback: plain shuffle (should never happen with 4 classes)
    out = list(seq)
    random.shuffle(out)
    return out


def make_run_sequence(prev_seq=None):
    forbidden = prev_seq[-1] if (P['avoid_repeats'] and prev_seq) else None
    seq = CLASSES * P['n_repeats']
    if P['avoid_repeats']:
        for _ in range(20):
            result = _shuffle_no_repeats(seq, forbidden_first=forbidden)
            if result != prev_seq:
                return result
        return _shuffle_no_repeats(seq, forbidden_first=forbidden)
    else:
        for _ in range(1000):
            random.shuffle(seq)
            if seq != prev_seq:
                return seq
        return seq


# ===========================================================================
#  SETUP
# ===========================================================================
mon = monitors.Monitor('expMonitor')
win = visual.Window(size=P['win_size'], fullscr=P['fullscreen'], screen=P['screen'],
                    color=P['bg_color'], units='height', monitor=mon, allowGUI=False)

globalClock = core.Clock()
frameClock  = core.Clock()
MARKERS = []
CUR = {'run': 0, 'trial': 0}

fixation  = make_cross([0.6, 0.6, 0.6])   # light grey
rot_cross = make_cross([0.2, 0.2, 0.2])   # darker grey
rot_cross.ori = 0.0

placeholder = visual.TextStim(win, text='', color=P['fg_color'], height=0.1, units='height')
txt = visual.TextStim(win, text='', color=P['fg_color'], height=0.038,
                      wrapWidth=1.4, units='height')
pd = visual.Rect(win, width=0.12, height=0.12, fillColor='white', lineColor='white',
                 pos=(0.78, -0.42), units='height')

# circle-mode stims (used when P['circle_mode'] is True)
breath_circle = visual.Circle(win, radius=0.05, fillColor=[0.6, 0.6, 0.6],
                               lineColor=None, units='height')
# static ring at r_ring — always visible in circle mode, marks the MI threshold
mi_ring       = visual.Circle(win, radius=P['circle_r_ring'], fillColor=None,
                               lineColor=[-0.3, -0.3, -0.3],  # dark grey
                               lineWidth=P['circle_line_w'], units='height')

IMAGE_PATHS, IMG_STIMS = {}, {}
for c in CLASSES:
    p = os.path.join(P['images_dir'], IMAGE_FILES[c])
    if os.path.isfile(p):
        IMAGE_PATHS[c] = p
        IMG_STIMS[c]   = visual.ImageStim(win, image=p, units='height')
    else:
        IMAGE_PATHS[c] = None
        IMG_STIMS[c]   = None
        print('[INFO] Image "%s" not found (%s) -> text placeholder.' % (c, p))

trig = TriggerBox(P['trigger_backend'], P['parallel_address'])

if P['rng_seed'] is not None:
    random.seed(int(P['rng_seed']))
else:
    random.seed()  # force re-seed from os.urandom, overrides any app-level fixed seed


# ===========================================================================
#  MAIN
# ===========================================================================
try:
    _n_trials    = 4 * P['n_repeats']
    _trial_s     = TICKS_PER_TRIAL * overlap_period
    _run_s       = _n_trials * _trial_s
    _total_s     = _run_s * P['n_runs']

    def _fmt(s):
        m, s2 = divmod(int(s), 60)
        return '%d:%02d' % (m, s2) if m else '%ds' % s2

    if P['circle_mode']:
        _instr = (
            "Motor Imagery Experiment\n\n"
            "Keep your eyes on the circle in the centre.\n\n"
            "When the circle shrinks, a picture will appear\n"
            "showing what to imagine:\n"
            "   REST  /  LEFT hand  /  RIGHT hand  /  FEET\n\n"
            "Imagine the movement ONLY while the circle is within the ring.\n"
            "Stop imagining when the circle grows back out.\n\n"
            "Trial: %.1f s   |   Run: %s (%d trials)   |   Total: %s (%d runs)\n\n"
            "Press SPACE to begin."
            % (_trial_s, _fmt(_run_s), _n_trials, _fmt(_total_s), P['n_runs'])
        )
    else:
        _instr = (
            "Motor Imagery Experiment\n\n"
            "Keep your eyes on the cross in the centre.\n\n"
            "When a picture appears it tells you what to imagine:\n"
            "   REST  /  LEFT hand  /  RIGHT hand  /  FEET\n\n"
            "Start imagining at the first TICK after the picture appears.\n"
            "Stop imagining at the next TICK.\n"
            "One TICK = when the rotating cross aligns with the fixed cross.\n\n"
            "Trial: %.1f s   |   Run: %s (%d trials)   |   Total: %s (%d runs)\n\n"
            "Press SPACE to begin."
            % (_trial_s, _fmt(_run_s), _n_trials, _fmt(_total_s), P['n_runs'])
        )
    show_text(_instr)

    # --- one-time fade-in of cross / circle over lead_in seconds ----------
    rot_cross.opacity = 0.0
    if P['circle_mode']:
        breath_circle.opacity = 0.0
        mi_ring.opacity = 0.0
    _fade_clock = core.Clock()
    while _fade_clock.getTime() < P['lead_in']:
        fade = min(_fade_clock.getTime() / P['lead_in'], 1.0)
        if P['circle_mode']:
            breath_circle.opacity = fade
            mi_ring.opacity = fade
        else:
            rot_cross.opacity = fade
        draw_scene()
        flip()
        check_quit()
    rot_cross.opacity = 1.0
    if P['circle_mode']:
        breath_circle.opacity = 1.0
        mi_ring.opacity = 1.0

    prev_seq = None
    for run in range(1, P['n_runs'] + 1):
        CUR['run'] = run
        seq = make_run_sequence(prev_seq)
        prev_seq = seq[:]

        if run > 1:
            if P['circle_mode']:
                show_text(
                    "Break.\n\n"
                    "Run %d of %d coming up.\n\n"
                    "Watch the circle — imagine the movement\n"
                    "only while the picture is visible.\n\n"
                    "Press SPACE when ready." % (run, P['n_runs'])
                )
            else:
                show_text(
                    "Break.\n\n"
                    "Run %d of %d coming up.\n\n"
                    "Watch the cross — imagine the movement\n"
                    "only while the picture is visible.\n\n"
                    "Press SPACE when ready." % (run, P['n_runs'])
                )

        globalClock.reset()
        frameClock.reset()
        win.callOnFlip(trig.send, TRIG['run_start'], 'run_start')
        draw_scene()
        flip()

        phase_cooldown()

        for i, cls in enumerate(seq, start=1):
            CUR['trial'] = i
            run_trial(cls)

        win.callOnFlip(trig.send, TRIG['run_end'], 'run_end')
        draw_scene()
        flip()
        core.wait(0.3)

    show_text("All runs complete.\n\nThank you!\n\nPress SPACE to exit.")

except QuitExperiment:
    print('[INFO] Experiment aborted by user (escape).')

finally:
    save_markers()
    try:
        win.close()
    except Exception:
        pass

core.quit()