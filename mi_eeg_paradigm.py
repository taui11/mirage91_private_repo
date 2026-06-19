#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LABRECORDER stream finden
Timing


============================================================================
 Motor-Imagery cue paradigm for EEG  --  PsychoPy (Coder / standalone .py)
============================================================================

ALL TIMING IS DRIVEN BY CROSS OVERLAPS.

The second cross rotates continuously.  Every time it aligns with the
fixed cross (every 90 deg of rotation) counts as one "overlap tick".
The overlap period = tick_time  seconds  (rot_speed = 90/tick_time deg/s).

Timeline of ONE TRIAL  (overlap ticks marked with |):

   [0 .. t_fadein)                fade IN   (precue trigger)
   [t_fadein .. t_fadein+t_stay)  stay      (MI trigger)
   [t_fadein+t_stay .. total)     fade OUT
   n_pause ticks after fade-out:  pause, cross only (pause trigger)

   Before the first trial: n_cooldown ticks of just the crosses
   so the participant can settle.

   Triggers fire at phase transitions, class-specific for precue and MI
   so you can epoch by condition.

   n_runs  runs,  each with  n_repeats * 4 classes  trials.

HOW TO RUN
  1. Install PsychoPy  (standalone app, or  pip install psychopy).
  2. Put 4 images in an "images/" folder next to this file.
     Missing images -> text placeholder so you can test timing.
  3. PsychoPy Coder -> Open -> Run,  or  python mi_eeg_paradigm.py
  4. ESC to abort (markers saved).

EEG MARKERS
  trigger_backend:  'lsl' | 'parallel' | 'none' (CSV log only).
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
    'n_runs'          : 2,
    'n_repeats'       : 1,        # per class per run -> 4*10 = 40 trials/run

    # --- timing (all in overlap ticks; 1 tick = tick_time seconds) ---
    # Trial: fade-in -> stay -> fade-out -> pause (cross only)
    't_fadein'        : 0.5,       # ticks for picture to fade in
    't_stay'          : 1.0,       # ticks picture stays fully visible
    't_fadeout'       : 0.5,       # ticks for picture to fade out
    'n_pause'         : 1,         # whole ticks of pause (cross only) after fade-out
    'n_cooldown'      : 1,         # ticks of just-cross before first trial

    # --- cross appearance ---
    'cross_size'      : 0.25,      # half-length of each arm  (height units)
    'cross_thickness' : 0.003,     # half-width of each bar   (height units)

    # --- rotating cross ---
    'tick_time'       : 1.5,       # seconds per overlap tick  (rot_speed = 90/tick_time deg/s)
    'overlap_tol'     : 3.0,       # degrees for overlap detection

    # --- picture appearance ---
    'bigpicture'      : False,     # True  = faint overlay;  False = small cardinal positions
    'overlay_opacity' : 0.20,      # peak opacity when bigpicture = True
    'small_size'      : (0.225, 0.225),
    # big_size and small_size are per-class; see BIG_SIZE / SMALL_SIZE below

    # --- hardware / display ---
    'fullscreen'      : True,
    'screen'          : 0,
    'win_size'        : (1920, 1080),
    'bg_color'        : 'black',
    'fg_color'        : 'white',
    'use_photodiode'  : False,
    'trigger_backend' : 'none',    # 'none' | 'lsl' | 'parallel'
    'parallel_address': 0x0378,

    # --- misc ---
    'rng_seed'        : None,
    'images_dir'      : 'images',
    'avoid_repeats'   : False,
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
#  STARTUP DIALOG
# ===========================================================================
expinfo = {
    'participant'    : P['participant'],
    'session'        : P['session'],
    'n_runs'         : P['n_runs'],
    'n_repeats'      : P['n_repeats'],
    't_fadein'       : P['t_fadein'],
    't_stay'         : P['t_stay'],
    't_fadeout'      : P['t_fadeout'],
    'n_pause'        : P['n_pause'],
    'n_cooldown'     : P['n_cooldown'],
    'tick_time'      : P['tick_time'],
    'bigpicture'     : P['bigpicture'],
    'avoid_repeats'  : P['avoid_repeats'],
    'fullscreen'     : P['fullscreen'],
    'trigger_backend': ['none', 'lsl', 'parallel'],
}
_order = ['participant', 'session', 'n_runs', 'n_repeats',
          't_fadein', 't_stay', 't_fadeout', 'n_pause', 'n_cooldown', 'tick_time',
          'bigpicture', 'avoid_repeats', 'fullscreen', 'trigger_backend']
_dlg = gui.DlgFromDict(expinfo, title='MI EEG paradigm', order=_order)
if not _dlg.OK:
    core.quit()
P.update(expinfo)

# --- force correct types (DlgFromDict returns strings) ---
for k in ('n_runs', 'n_repeats', 'n_pause', 'n_cooldown'):
    P[k] = int(P[k])
for k in ('tick_time', 't_fadein', 't_stay', 't_fadeout',
          'overlap_tol', 'cross_size', 'cross_thickness',
          'overlay_opacity', 'lead_in'):
    P[k] = float(P[k])
if isinstance(P['bigpicture'], str):
    P['bigpicture'] = P['bigpicture'] in ('True', 'true', '1')
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


def draw_scene(extra=None, photodiode_on=False):
    advance_rotation()
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
    """Wait for n overlap ticks (rising edges), drawing cross each frame."""
    for _ in range(n):
        wait_for_overlap()


def phase_cooldown():
    """Spin the crosses for n_cooldown ticks so the participant can get ready."""
    win.callOnFlip(trig.send, TRIG['pause'], 'cooldown')
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


# Module-level mutable for tick-start timestamp (avoids a global statement)
_tick_start = [0.0]


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
    peak      = P['overlay_opacity'] if P['bigpicture'] else 1.0
    t_fi      = P['t_fadein']  * overlap_period   # seconds
    t_st      = P['t_stay']    * overlap_period
    t_fo      = P['t_fadeout'] * overlap_period
    t_total   = t_fi + t_st + t_fo

    stim = _get_stim(cls)
    stim.opacity = 0.0

    mi_fired = False

    # wait for the next overlap tick, then start the picture clock
    win.callOnFlip(trig.send, TRIG['precue'][cls], 'precue_%s' % cls)
    _wait_for_overlap_tracked()
    pic_clock = core.Clock()

    # ---- Picture phase: runs for t_total seconds, frame by frame -------------
    while True:
        was_in = _is_overlapping()
        elapsed = pic_clock.getTime()

        if elapsed < t_fi:
            stim.opacity = peak * (elapsed / t_fi) if t_fi > 0 else peak
        elif elapsed < t_fi + t_st:
            if not mi_fired:
                win.callOnFlip(trig.send, TRIG['mi'][cls], 'mi_%s' % cls)
                mi_fired = True
            stim.opacity = peak
        elif elapsed < t_total:
            remaining = t_total - elapsed
            stim.opacity = peak * (remaining / t_fo) if t_fo > 0 else 0.0
        else:
            stim.opacity = 0.0

        show_pic = stim.opacity > 0.01
        draw_scene(extra=stim if show_pic else None, photodiode_on=show_pic)
        flip()
        check_quit()

        # once picture is done, wait for next overlap then hand off to pause
        if elapsed >= t_total:
            if not was_in and _is_overlapping():
                _tick_start[0] = globalClock.getTime()
                break

    stim.opacity = 0.0

    # ---- Pause ticks: cross only ----------------------------------------------
    win.callOnFlip(trig.send, TRIG['pause'], 'pause')
    _wait_n_ticks(P['n_pause'])


def _get_stim(cls):
    """Return the image stim (or text placeholder) for a class, configured."""
    pos = (0.0, 0.0) if P['bigpicture'] else SMALL_POS[cls]
    if IMG_STIMS[cls] is not None:
        stim = IMG_STIMS[cls]
        stim.size = BIG_SIZE[cls] if P['bigpicture'] else SMALL_SIZE[cls]
        stim.pos  = pos
    else:
        stim = placeholder
        stim.text   = cls.upper()
        stim.height = 0.15 if P['bigpicture'] else 0.09
        stim.pos    = pos
    return stim


# --- trial sequence --------------------------------------------------------
def _limit_runs(seq, max_run=2):
    for _ in range(1000):
        ok, run = True, 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                run += 1
                if run > max_run:
                    ok = False
                    break
            else:
                run = 1
        if ok:
            return seq
        random.shuffle(seq)
    return seq


def make_run_sequence(prev_seq=None):
    seq = CLASSES * P['n_repeats']
    for _ in range(1000):
        random.shuffle(seq)
        if P['avoid_repeats']:
            seq = _limit_runs(seq, max_run=1)
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
txt = visual.TextStim(win, text='', color=P['fg_color'], height=0.05,
                      wrapWidth=1.4, units='height')
pd = visual.Rect(win, width=0.12, height=0.12, fillColor='white', lineColor='white',
                 pos=(0.78, -0.42), units='height')

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
    show_text(
        "Motor Imagery Experiment\n\n"
        "Keep your eyes on the cross in the centre.\n\n"
        "When a picture appears it tells you what to imagine:\n"
        "   REST  /  LEFT hand  /  RIGHT hand  /  FEET\n\n"
        "After the picture disappears, imagine that movement\n"
        "until the next picture appears.\n\n"
        "Each trial = %d overlap ticks  (%.1f s)\n"
        "%d trials / run,  %d runs\n\n"
        "Press SPACE to begin."
        % (TICKS_PER_TRIAL, TICKS_PER_TRIAL * overlap_period,
           4 * P['n_repeats'], P['n_runs'])
    )

    # --- one-time fade-in of the spinning cross over lead_in seconds ----------
    rot_cross.opacity = 0.0
    _fade_clock = core.Clock()
    while _fade_clock.getTime() < P['lead_in']:
        rot_cross.opacity = min(_fade_clock.getTime() / P['lead_in'], 1.0)
        draw_scene()
        flip()
        check_quit()
    rot_cross.opacity = 1.0

    prev_seq = None
    for run in range(1, P['n_runs'] + 1):
        CUR['run'] = run
        seq = make_run_sequence(prev_seq)
        prev_seq = seq[:]

        if run > 1:
            show_text("Break.\n\nRun %d of %d coming up.\n\n"
                      "Press SPACE when ready." % (run, P['n_runs']))

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