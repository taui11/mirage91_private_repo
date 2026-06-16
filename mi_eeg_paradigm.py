#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================================
 Motor-Imagery cue paradigm for EEG  --  PsychoPy (Coder / standalone .py)
============================================================================

ALL TIMING IS DRIVEN BY CROSS OVERLAPS.

The second cross rotates continuously.  Every time it aligns with the
fixed cross (every 90 deg of rotation) counts as one "overlap tick".
The overlap period = 90 / rot_speed  seconds.

Timeline of ONE TRIAL  (overlap ticks marked with |):

   |--- n_pause ---|--- n_precue ---|--- n_mi ---|
        PAUSE          FADE IN        MI + FADE OUT
     (rest, just       (picture       (imagine! picture
      the crosses)     fades in)      fades out)

   ... then the cycle repeats for the next trial.

   Triggers fire at the onset of each phase (first frame), class-specific
   for precue and MI so you can epoch by condition.

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

I changed some variables in the code so keep them as they are now.

CHANGES that need to be implemented:

-small picture mode each activity picture different cardinal direction left=left right=right rest=top feet=bottom
-Pictures flicker at the start should be smooth transition also fadein should start a changeable tickspace after overlap same for fade out
-Fade in should start 0.5 ticks after cossing then the picture should stay 0.5 ticks and fadeout the other 0.5 ticks (0 nothing 0.5 fadein(precue) 1 stay+MI 1.5 fadeout+MI) 2 pause 3 repeat)
-let the cross do n_cooldown ticks of nothing before starting so the participant can get ready
-what does the backend trigger do
-what are the markers? how does t_sec work?


"""

from psychopy import visual, core, event, gui, monitors, logging
import os, csv, random, datetime, math

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

    # --- timing (in overlap counts) ---
    'n_pause'         : 1,         # overlap ticks of rest/pause
    'n_precue'        : 1,         # overlap ticks for fade-in
    'n_mi'            : 1,         # overlap ticks of MI (picture fades out)

    # --- cross appearance ---
    'cross_size'      : 0.25,      # half-length of each arm  (height units)
    'cross_thickness' : 0.003,     # half-width of each bar   (height units)

    # --- rotating cross ---
    'rot_speed'       : 60.0,      # deg/s  (overlap period = 90/speed s)
    'overlap_tol'     : 3.0,       # degrees for overlap detection

    # --- picture appearance ---
    'bigpicture'      : True,     # True  = faint overlay;  False = small left
    'overlay_opacity' : 0.35,      # peak opacity when bigpicture = True
    'big_size'        : (0.6, 0.6),
    'small_size'      : (0.45, 0.45),
    'small_pos'       : (-0.35, 0.0),

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
    'participant'     : P['participant'],
    'session'         : P['session'],
    'n_runs'          : P['n_runs'],
    'n_repeats'       : P['n_repeats'],
    'n_pause'         : P['n_pause'],
    'n_precue'        : P['n_precue'],
    'n_mi'            : P['n_mi'],
    'rot_speed'       : P['rot_speed'],
    'bigpicture'      : P['bigpicture'],
    'fullscreen'      : P['fullscreen'],
    'trigger_backend' : ['none', 'lsl', 'parallel'],
}
_order = ['participant', 'session', 'n_runs', 'n_repeats',
          'n_pause', 'n_precue', 'n_mi', 'rot_speed',
          'bigpicture', 'fullscreen', 'trigger_backend']
_dlg = gui.DlgFromDict(expinfo, title='MI EEG paradigm', order=_order)
if not _dlg.OK:
    core.quit()
P.update(expinfo)

# --- force correct types (DlgFromDict returns strings) ---
for k in ('n_runs', 'n_repeats', 'n_pause', 'n_precue', 'n_mi'):
    P[k] = int(P[k])
for k in ('rot_speed', 'overlap_tol', 'cross_size', 'cross_thickness',
          'overlay_opacity', 'lead_in'):
    P[k] = float(P[k])
if isinstance(P['bigpicture'], str):
    P['bigpicture'] = P['bigpicture'] in ('True', 'true', '1')
if isinstance(P['fullscreen'], str):
    P['fullscreen'] = P['fullscreen'] in ('True', 'true', '1')

overlap_period = 90.0 / P['rot_speed']   # seconds between overlaps
print('[INFO] Overlap period = %.3f s' % overlap_period)
print('[INFO] Trial duration = %d overlaps = %.1f s'
      % (P['n_pause'] + P['n_precue'] + P['n_mi'],
         (P['n_pause'] + P['n_precue'] + P['n_mi']) * overlap_period))
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
        draw_scene(); flip(); check_quit()
    # now wait to enter
    while not _is_overlapping():
        draw_scene(); flip(); check_quit()


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

def phase_pause():
    """Rest / pause: just the spinning crosses for n_pause overlaps."""
    win.callOnFlip(trig.send, TRIG['pause'], 'pause')
    for _ in range(P['n_pause']):
        wait_for_overlap()


def phase_precue(cls):
    """Picture fades in over n_precue overlaps."""
    peak = P['overlay_opacity'] if P['bigpicture'] else 1.0
    stim = _get_stim(cls)

    win.callOnFlip(trig.send, TRIG['precue'][cls], 'precue_%s' % cls)

    n = P['n_precue']
    for i in range(n):
        # opacity ramps linearly:  overlap 0 -> 1/n,  overlap n-1 -> n/n
        op_start = peak * (i / n)
        op_end   = peak * ((i + 1) / n)
        # draw frames until the next overlap, interpolating opacity
        t0 = globalClock.getTime()
        dur_est = overlap_period          # approximate
        while True:
            was_in = _is_overlapping()
            draw_scene(extra=stim, photodiode_on=True)
            flip(); check_quit()
            # interpolate opacity based on time within this overlap interval
            elapsed = globalClock.getTime() - t0
            frac = min(elapsed / dur_est, 1.0) if dur_est > 0 else 1.0
            stim.opacity = op_start + (op_end - op_start) * frac
            # detect next overlap (rising edge)
            if not was_in and _is_overlapping():
                break


def phase_mi(cls):
    """MI while picture fades out over n_mi overlaps."""
    peak = P['overlay_opacity'] if P['bigpicture'] else 1.0
    stim = _get_stim(cls)
    stim.opacity = peak                   # start fully visible

    win.callOnFlip(trig.send, TRIG['mi'][cls], 'mi_%s' % cls)

    n = P['n_mi']
    for i in range(n):
        op_start = peak * (1.0 - i / n)
        op_end   = peak * (1.0 - (i + 1) / n)
        t0 = globalClock.getTime()
        dur_est = overlap_period
        while True:
            was_in = _is_overlapping()
            show_pic = stim.opacity > 0.01
            draw_scene(extra=stim if show_pic else None, photodiode_on=show_pic)
            flip(); check_quit()
            elapsed = globalClock.getTime() - t0
            frac = min(elapsed / dur_est, 1.0) if dur_est > 0 else 1.0
            stim.opacity = op_start + (op_end - op_start) * frac
            if not was_in and _is_overlapping():
                break
    stim.opacity = 0.0


def _get_stim(cls):
    """Return the image stim (or text placeholder) for a class, configured."""
    if IMG_STIMS[cls] is not None:
        stim = IMG_STIMS[cls]
        stim.size = P['big_size'] if P['bigpicture'] else P['small_size']
        stim.pos  = (0.0, 0.0) if P['bigpicture'] else P['small_pos']
    else:
        stim = placeholder
        stim.text   = cls.upper()
        stim.height = 0.15 if P['bigpicture'] else 0.09
        stim.pos    = (0.0, 0.0) if P['bigpicture'] else P['small_pos']
    return stim


# --- trial sequence --------------------------------------------------------
def _limit_runs(seq, max_run=2):
    for _ in range(1000):
        ok, run = True, 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                run += 1
                if run > max_run:
                    ok = False; break
            else:
                run = 1
        if ok:
            return seq
        random.shuffle(seq)
    return seq


def make_run_sequence():
    seq = CLASSES * P['n_repeats']
    random.shuffle(seq)
    if P['avoid_repeats']:
        seq = _limit_runs(seq, max_run=2)
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

fixation  = make_cross(P['fg_color'])
rot_cross = make_cross(P['fg_color'])
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


# ===========================================================================
#  MAIN
# ===========================================================================
try:
    _ticks = P['n_pause'] + P['n_precue'] + P['n_mi']
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
        % (_ticks, _ticks * overlap_period,
           4 * P['n_repeats'], P['n_runs'])
    )

    for run in range(1, P['n_runs'] + 1):
        CUR['run'] = run
        seq = make_run_sequence()

        if run > 1:
            show_text("Break.\n\nRun %d of %d coming up.\n\n"
                      "Press SPACE when ready." % (run, P['n_runs']))

        globalClock.reset()
        frameClock.reset()
        win.callOnFlip(trig.send, TRIG['run_start'], 'run_start')
        draw_scene(); flip()
        core.wait(P['lead_in'])

        for i, cls in enumerate(seq, start=1):
            CUR['trial'] = i
            phase_pause()
            phase_precue(cls)
            phase_mi(cls)

        win.callOnFlip(trig.send, TRIG['run_end'], 'run_end')
        draw_scene(); flip()
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