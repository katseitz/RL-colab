#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Wed Mar 12 12:06:01 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from probability_sequence_code
import random
import csv
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'RL_reversal'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': 'ses-1',
    'practice': [0,1],
    'restart_from_run': [None, 2,3],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1728, 1117]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/katharinaseitz/Documents/projects/RL-colab/reinforcement_learning_volatility_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[-1,-1,-1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1,-1,-1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('cue_resp_lp') is None:
        # initialise cue_resp_lp
        cue_resp_lp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='cue_resp_lp',
        )
    if deviceManager.getDevice('cue_resp_rp') is None:
        # initialise cue_resp_rp
        cue_resp_rp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='cue_resp_rp',
        )
    if deviceManager.getDevice('advance_to_runs') is None:
        # initialise advance_to_runs
        advance_to_runs = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advance_to_runs',
        )
    if deviceManager.getDevice('advance_press') is None:
        # initialise advance_press
        advance_press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advance_press',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('cue_resp') is None:
        # initialise cue_resp
        cue_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='cue_resp',
        )
    if deviceManager.getDevice('all_keys') is None:
        # initialise all_keys
        all_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='all_keys',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "prac_left" ---
    left_box_lp = visual.ImageStim(
        win=win,
        name='left_box_lp', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_box_lp = visual.ImageStim(
        win=win,
        name='right_box_lp', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    cue_resp_lp = keyboard.Keyboard(deviceName='cue_resp_lp')
    prac_left_text = visual.TextStim(win=win, name='prac_left_text',
        text='Practice selecting the right box by pushing the "1" key on your keyboard. ',
        font='Arial',
        pos=(0, .2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "prac_left_response" ---
    left_box_lr = visual.ImageStim(
        win=win,
        name='left_box_lr', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    right_box_lr = visual.ImageStim(
        win=win,
        name='right_box_lr', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "prac_right" ---
    left_box_rp = visual.ImageStim(
        win=win,
        name='left_box_rp', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_box_rp = visual.ImageStim(
        win=win,
        name='right_box_rp', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    cue_resp_rp = keyboard.Keyboard(deviceName='cue_resp_rp')
    prac_right_text = visual.TextStim(win=win, name='prac_right_text',
        text='Practice selecting the right box by pushing the "2" key on your keyboard. ',
        font='Arial',
        pos=(0, .2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "prac_right_response" ---
    left_box_rr = visual.ImageStim(
        win=win,
        name='left_box_rr', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    right_box_rr = visual.ImageStim(
        win=win,
        name='right_box_rr', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "summary_instructions" ---
    summary_text = visual.TextStim(win=win, name='summary_text',
        text='1. In the game, there will be two boxes, but only one is magical.\n\n2. Sometimes, the magical box switches sides!\n\n3. Sometimes, even the magical box does not have a coin.\n\n\nTry to collect all the coins!\n\nPress the space bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from define_switches_code
    #number correct for the switch occurs
    num_switch_list = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    #now shuffle the list
    random.shuffle(num_switch_list)
    # Run 'Begin Experiment' code from probability_sequence_code
    sequence = []
    exp_correct = 0 #number of correct choices made in the experiment
    
    
    #TODO: pull this from experiment variables.
    number_of_trials = 150
    
    while len(sequence) < number_of_trials:
        #generate a probabilty list at 75p by alternating
        #ten trials at 80p and ten trials 70p.
        sequence_80 = [0] * 2 + [1] * 8
        random.shuffle(sequence_80)
        sequence_70 = [0] * 3 + [1] * 7
        random.shuffle(sequence_70)
        sequence = sequence + sequence_80 + sequence_70
        #TODO: output this sequence to a .csv that gets saved.
    
    
    with open('test.csv', 'w', newline='') as myfile:
         wr = csv.writer(myfile)
         wr.writerow(sequence)
    advance_to_runs = keyboard.Keyboard(deviceName='advance_to_runs')
    # Run 'Begin Experiment' code from restart_mid_exp_code
    if expInfo["restart_from_run"]:
        #if 3, have 1 run, if 2, have 2 runs
        num_runs = 4 - expInfo["restart_from_run"] 
    else:
        num_runs = 3
    
    # --- Initialize components for Routine "get_ready" ---
    get_ready_text = visual.TextStim(win=win, name='get_ready_text',
        text='Get Ready!\n\nPress the space bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advance_press = keyboard.Keyboard(deviceName='advance_press')
    
    # --- Initialize components for Routine "scanner_trigger" ---
    scanner_text = visual.TextStim(win=win, name='scanner_text',
        text='waiting for scanner\n\nPress "t" to emulate scanner trigger.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "cue" ---
    left_box = visual.ImageStim(
        win=win,
        name='left_box', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_box = visual.ImageStim(
        win=win,
        name='right_box', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    cue_resp = keyboard.Keyboard(deviceName='cue_resp')
    # Run 'Begin Experiment' code from good_side_code
    good_side = "right"
    #counter = 0
    num_rewarded = 0
    num_switch = 0
    
    all_keys = keyboard.Keyboard(deviceName='all_keys')
    
    # --- Initialize components for Routine "cue_response" ---
    left_box_response = visual.ImageStim(
        win=win,
        name='left_box_response', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    right_box_response = visual.ImageStim(
        win=win,
        name='right_box_response', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "outcome" ---
    outcome_left_box = visual.ImageStim(
        win=win,
        name='outcome_left_box', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    outcome_right_box = visual.ImageStim(
        win=win,
        name='outcome_right_box', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "fixation" ---
    fixation_cross = visual.TextStim(win=win, name='fixation_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "run_feedback" ---
    coins_won_text = visual.TextStim(win=win, name='coins_won_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "leftover_time_break" ---
    fixation_end = visual.TextStim(win=win, name='fixation_end',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "end_exp" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    practice_loop = data.TrialHandler2(
        name='practice_loop',
        nReps=expInfo["practice"], 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(practice_loop)  # add the loop to the experiment
    thisPractice_loop = practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
    if thisPractice_loop != None:
        for paramName in thisPractice_loop:
            globals()[paramName] = thisPractice_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_loop in practice_loop:
        currentLoop = practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_loop.rgb)
        if thisPractice_loop != None:
            for paramName in thisPractice_loop:
                globals()[paramName] = thisPractice_loop[paramName]
        
        # --- Prepare to start Routine "prac_left" ---
        # create an object to store info about Routine prac_left
        prac_left = data.Routine(
            name='prac_left',
            components=[left_box_lp, right_box_lp, cue_resp_lp, prac_left_text],
        )
        prac_left.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for cue_resp_lp
        cue_resp_lp.keys = []
        cue_resp_lp.rt = []
        _cue_resp_lp_allKeys = []
        # store start times for prac_left
        prac_left.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_left.tStart = globalClock.getTime(format='float')
        prac_left.status = STARTED
        thisExp.addData('prac_left.started', prac_left.tStart)
        prac_left.maxDuration = None
        # keep track of which components have finished
        prac_leftComponents = prac_left.components
        for thisComponent in prac_left.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_left" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        prac_left.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *left_box_lp* updates
            
            # if left_box_lp is starting this frame...
            if left_box_lp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_box_lp.frameNStart = frameN  # exact frame index
                left_box_lp.tStart = t  # local t and not account for scr refresh
                left_box_lp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_box_lp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_box_lp.started')
                # update status
                left_box_lp.status = STARTED
                left_box_lp.setAutoDraw(True)
            
            # if left_box_lp is active this frame...
            if left_box_lp.status == STARTED:
                # update params
                pass
            
            # *right_box_lp* updates
            
            # if right_box_lp is starting this frame...
            if right_box_lp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_box_lp.frameNStart = frameN  # exact frame index
                right_box_lp.tStart = t  # local t and not account for scr refresh
                right_box_lp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_box_lp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_box_lp.started')
                # update status
                right_box_lp.status = STARTED
                right_box_lp.setAutoDraw(True)
            
            # if right_box_lp is active this frame...
            if right_box_lp.status == STARTED:
                # update params
                pass
            
            # *cue_resp_lp* updates
            waitOnFlip = False
            
            # if cue_resp_lp is starting this frame...
            if cue_resp_lp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_resp_lp.frameNStart = frameN  # exact frame index
                cue_resp_lp.tStart = t  # local t and not account for scr refresh
                cue_resp_lp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_resp_lp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_resp_lp.started')
                # update status
                cue_resp_lp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(cue_resp_lp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(cue_resp_lp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if cue_resp_lp.status == STARTED and not waitOnFlip:
                theseKeys = cue_resp_lp.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
                _cue_resp_lp_allKeys.extend(theseKeys)
                if len(_cue_resp_lp_allKeys):
                    cue_resp_lp.keys = _cue_resp_lp_allKeys[0].name  # just the first key pressed
                    cue_resp_lp.rt = _cue_resp_lp_allKeys[0].rt
                    cue_resp_lp.duration = _cue_resp_lp_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *prac_left_text* updates
            
            # if prac_left_text is starting this frame...
            if prac_left_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_left_text.frameNStart = frameN  # exact frame index
                prac_left_text.tStart = t  # local t and not account for scr refresh
                prac_left_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_left_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_left_text.started')
                # update status
                prac_left_text.status = STARTED
                prac_left_text.setAutoDraw(True)
            
            # if prac_left_text is active this frame...
            if prac_left_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_left.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_left.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_left" ---
        for thisComponent in prac_left.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_left
        prac_left.tStop = globalClock.getTime(format='float')
        prac_left.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_left.stopped', prac_left.tStop)
        # check responses
        if cue_resp_lp.keys in ['', [], None]:  # No response was made
            cue_resp_lp.keys = None
        practice_loop.addData('cue_resp_lp.keys',cue_resp_lp.keys)
        if cue_resp_lp.keys != None:  # we had a response
            practice_loop.addData('cue_resp_lp.rt', cue_resp_lp.rt)
            practice_loop.addData('cue_resp_lp.duration', cue_resp_lp.duration)
        # the Routine "prac_left" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_left_response" ---
        # create an object to store info about Routine prac_left_response
        prac_left_response = data.Routine(
            name='prac_left_response',
            components=[left_box_lr, right_box_lr],
        )
        prac_left_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from prac_left_feedback
        #initialize selection indicator
        pr_selection_indicator = visual.Rect(
            win=win, name='polygon',
            width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
            ori=0.0, draggable=False, anchor='center',
            lineWidth=4.0,
            pos = (-0.3, -0.15),
            colorSpace='rgb', lineColor='white', fillColor=None,
            depth=-4.0, interpolate=True, 
            autoDraw = True)
         
        
        # store start times for prac_left_response
        prac_left_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_left_response.tStart = globalClock.getTime(format='float')
        prac_left_response.status = STARTED
        thisExp.addData('prac_left_response.started', prac_left_response.tStart)
        prac_left_response.maxDuration = None
        # keep track of which components have finished
        prac_left_responseComponents = prac_left_response.components
        for thisComponent in prac_left_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_left_response" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        prac_left_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from prac_left_feedback
            
                
            
            
            # *left_box_lr* updates
            
            # if left_box_lr is starting this frame...
            if left_box_lr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_box_lr.frameNStart = frameN  # exact frame index
                left_box_lr.tStart = t  # local t and not account for scr refresh
                left_box_lr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_box_lr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_box_lr.started')
                # update status
                left_box_lr.status = STARTED
                left_box_lr.setAutoDraw(True)
            
            # if left_box_lr is active this frame...
            if left_box_lr.status == STARTED:
                # update params
                pass
            
            # if left_box_lr is stopping this frame...
            if left_box_lr.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > left_box_lr.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    left_box_lr.tStop = t  # not accounting for scr refresh
                    left_box_lr.tStopRefresh = tThisFlipGlobal  # on global time
                    left_box_lr.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_box_lr.stopped')
                    # update status
                    left_box_lr.status = FINISHED
                    left_box_lr.setAutoDraw(False)
            
            # *right_box_lr* updates
            
            # if right_box_lr is starting this frame...
            if right_box_lr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_box_lr.frameNStart = frameN  # exact frame index
                right_box_lr.tStart = t  # local t and not account for scr refresh
                right_box_lr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_box_lr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_box_lr.started')
                # update status
                right_box_lr.status = STARTED
                right_box_lr.setAutoDraw(True)
            
            # if right_box_lr is active this frame...
            if right_box_lr.status == STARTED:
                # update params
                pass
            
            # if right_box_lr is stopping this frame...
            if right_box_lr.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > right_box_lr.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    right_box_lr.tStop = t  # not accounting for scr refresh
                    right_box_lr.tStopRefresh = tThisFlipGlobal  # on global time
                    right_box_lr.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_box_lr.stopped')
                    # update status
                    right_box_lr.status = FINISHED
                    right_box_lr.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_left_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_left_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_left_response" ---
        for thisComponent in prac_left_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_left_response
        prac_left_response.tStop = globalClock.getTime(format='float')
        prac_left_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_left_response.stopped', prac_left_response.tStop)
        # Run 'End Routine' code from prac_left_feedback
        #turn off selection indicator
        pr_selection_indicator.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_left_response.maxDurationReached:
            routineTimer.addTime(-prac_left_response.maxDuration)
        elif prac_left_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "prac_right" ---
        # create an object to store info about Routine prac_right
        prac_right = data.Routine(
            name='prac_right',
            components=[left_box_rp, right_box_rp, cue_resp_rp, prac_right_text],
        )
        prac_right.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for cue_resp_rp
        cue_resp_rp.keys = []
        cue_resp_rp.rt = []
        _cue_resp_rp_allKeys = []
        # store start times for prac_right
        prac_right.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_right.tStart = globalClock.getTime(format='float')
        prac_right.status = STARTED
        thisExp.addData('prac_right.started', prac_right.tStart)
        prac_right.maxDuration = None
        # keep track of which components have finished
        prac_rightComponents = prac_right.components
        for thisComponent in prac_right.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_right" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        prac_right.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *left_box_rp* updates
            
            # if left_box_rp is starting this frame...
            if left_box_rp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_box_rp.frameNStart = frameN  # exact frame index
                left_box_rp.tStart = t  # local t and not account for scr refresh
                left_box_rp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_box_rp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_box_rp.started')
                # update status
                left_box_rp.status = STARTED
                left_box_rp.setAutoDraw(True)
            
            # if left_box_rp is active this frame...
            if left_box_rp.status == STARTED:
                # update params
                pass
            
            # *right_box_rp* updates
            
            # if right_box_rp is starting this frame...
            if right_box_rp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_box_rp.frameNStart = frameN  # exact frame index
                right_box_rp.tStart = t  # local t and not account for scr refresh
                right_box_rp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_box_rp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_box_rp.started')
                # update status
                right_box_rp.status = STARTED
                right_box_rp.setAutoDraw(True)
            
            # if right_box_rp is active this frame...
            if right_box_rp.status == STARTED:
                # update params
                pass
            
            # *cue_resp_rp* updates
            waitOnFlip = False
            
            # if cue_resp_rp is starting this frame...
            if cue_resp_rp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_resp_rp.frameNStart = frameN  # exact frame index
                cue_resp_rp.tStart = t  # local t and not account for scr refresh
                cue_resp_rp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_resp_rp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_resp_rp.started')
                # update status
                cue_resp_rp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(cue_resp_rp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(cue_resp_rp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if cue_resp_rp.status == STARTED and not waitOnFlip:
                theseKeys = cue_resp_rp.getKeys(keyList=['2'], ignoreKeys=["escape"], waitRelease=False)
                _cue_resp_rp_allKeys.extend(theseKeys)
                if len(_cue_resp_rp_allKeys):
                    cue_resp_rp.keys = _cue_resp_rp_allKeys[0].name  # just the first key pressed
                    cue_resp_rp.rt = _cue_resp_rp_allKeys[0].rt
                    cue_resp_rp.duration = _cue_resp_rp_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *prac_right_text* updates
            
            # if prac_right_text is starting this frame...
            if prac_right_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_right_text.frameNStart = frameN  # exact frame index
                prac_right_text.tStart = t  # local t and not account for scr refresh
                prac_right_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_right_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_right_text.started')
                # update status
                prac_right_text.status = STARTED
                prac_right_text.setAutoDraw(True)
            
            # if prac_right_text is active this frame...
            if prac_right_text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_right.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_right.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_right" ---
        for thisComponent in prac_right.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_right
        prac_right.tStop = globalClock.getTime(format='float')
        prac_right.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_right.stopped', prac_right.tStop)
        # check responses
        if cue_resp_rp.keys in ['', [], None]:  # No response was made
            cue_resp_rp.keys = None
        practice_loop.addData('cue_resp_rp.keys',cue_resp_rp.keys)
        if cue_resp_rp.keys != None:  # we had a response
            practice_loop.addData('cue_resp_rp.rt', cue_resp_rp.rt)
            practice_loop.addData('cue_resp_rp.duration', cue_resp_rp.duration)
        # the Routine "prac_right" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_right_response" ---
        # create an object to store info about Routine prac_right_response
        prac_right_response = data.Routine(
            name='prac_right_response',
            components=[left_box_rr, right_box_rr],
        )
        prac_right_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from prac_right_feedback
        #initialize selection indicator
        pr_selection_indicator = visual.Rect(
            win=win, name='polygon',
            width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
            ori=0.0, draggable=False, anchor='center',
            lineWidth=4.0,
            pos = (0.3, -0.15),
            colorSpace='rgb', lineColor='white', fillColor=None,
            depth=-4.0, interpolate=True, 
            autoDraw = True)
         
        
        # store start times for prac_right_response
        prac_right_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_right_response.tStart = globalClock.getTime(format='float')
        prac_right_response.status = STARTED
        thisExp.addData('prac_right_response.started', prac_right_response.tStart)
        prac_right_response.maxDuration = None
        # keep track of which components have finished
        prac_right_responseComponents = prac_right_response.components
        for thisComponent in prac_right_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "prac_right_response" ---
        # if trial has changed, end Routine now
        if isinstance(practice_loop, data.TrialHandler2) and thisPractice_loop.thisN != practice_loop.thisTrial.thisN:
            continueRoutine = False
        prac_right_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from prac_right_feedback
            
                
            
            
            # *left_box_rr* updates
            
            # if left_box_rr is starting this frame...
            if left_box_rr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_box_rr.frameNStart = frameN  # exact frame index
                left_box_rr.tStart = t  # local t and not account for scr refresh
                left_box_rr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_box_rr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_box_rr.started')
                # update status
                left_box_rr.status = STARTED
                left_box_rr.setAutoDraw(True)
            
            # if left_box_rr is active this frame...
            if left_box_rr.status == STARTED:
                # update params
                pass
            
            # if left_box_rr is stopping this frame...
            if left_box_rr.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > left_box_rr.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    left_box_rr.tStop = t  # not accounting for scr refresh
                    left_box_rr.tStopRefresh = tThisFlipGlobal  # on global time
                    left_box_rr.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_box_rr.stopped')
                    # update status
                    left_box_rr.status = FINISHED
                    left_box_rr.setAutoDraw(False)
            
            # *right_box_rr* updates
            
            # if right_box_rr is starting this frame...
            if right_box_rr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_box_rr.frameNStart = frameN  # exact frame index
                right_box_rr.tStart = t  # local t and not account for scr refresh
                right_box_rr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_box_rr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_box_rr.started')
                # update status
                right_box_rr.status = STARTED
                right_box_rr.setAutoDraw(True)
            
            # if right_box_rr is active this frame...
            if right_box_rr.status == STARTED:
                # update params
                pass
            
            # if right_box_rr is stopping this frame...
            if right_box_rr.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > right_box_rr.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    right_box_rr.tStop = t  # not accounting for scr refresh
                    right_box_rr.tStopRefresh = tThisFlipGlobal  # on global time
                    right_box_rr.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_box_rr.stopped')
                    # update status
                    right_box_rr.status = FINISHED
                    right_box_rr.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                prac_right_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_right_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_right_response" ---
        for thisComponent in prac_right_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_right_response
        prac_right_response.tStop = globalClock.getTime(format='float')
        prac_right_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_right_response.stopped', prac_right_response.tStop)
        # Run 'End Routine' code from prac_right_feedback
        #turn off selection indicator
        pr_selection_indicator.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_right_response.maxDurationReached:
            routineTimer.addTime(-prac_right_response.maxDuration)
        elif prac_right_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        thisExp.nextEntry()
        
    # completed expInfo["practice"] repeats of 'practice_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "summary_instructions" ---
    # create an object to store info about Routine summary_instructions
    summary_instructions = data.Routine(
        name='summary_instructions',
        components=[summary_text, advance_to_runs],
    )
    summary_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from probability_sequence_code
    num_gold_coins = 0
    # create starting attributes for advance_to_runs
    advance_to_runs.keys = []
    advance_to_runs.rt = []
    _advance_to_runs_allKeys = []
    # store start times for summary_instructions
    summary_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    summary_instructions.tStart = globalClock.getTime(format='float')
    summary_instructions.status = STARTED
    thisExp.addData('summary_instructions.started', summary_instructions.tStart)
    summary_instructions.maxDuration = None
    # keep track of which components have finished
    summary_instructionsComponents = summary_instructions.components
    for thisComponent in summary_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "summary_instructions" ---
    summary_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *summary_text* updates
        
        # if summary_text is starting this frame...
        if summary_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            summary_text.frameNStart = frameN  # exact frame index
            summary_text.tStart = t  # local t and not account for scr refresh
            summary_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(summary_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'summary_text.started')
            # update status
            summary_text.status = STARTED
            summary_text.setAutoDraw(True)
        
        # if summary_text is active this frame...
        if summary_text.status == STARTED:
            # update params
            pass
        
        # *advance_to_runs* updates
        waitOnFlip = False
        
        # if advance_to_runs is starting this frame...
        if advance_to_runs.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advance_to_runs.frameNStart = frameN  # exact frame index
            advance_to_runs.tStart = t  # local t and not account for scr refresh
            advance_to_runs.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advance_to_runs, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advance_to_runs.started')
            # update status
            advance_to_runs.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advance_to_runs.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advance_to_runs.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advance_to_runs.status == STARTED and not waitOnFlip:
            theseKeys = advance_to_runs.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _advance_to_runs_allKeys.extend(theseKeys)
            if len(_advance_to_runs_allKeys):
                advance_to_runs.keys = _advance_to_runs_allKeys[-1].name  # just the last key pressed
                advance_to_runs.rt = _advance_to_runs_allKeys[-1].rt
                advance_to_runs.duration = _advance_to_runs_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            summary_instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in summary_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "summary_instructions" ---
    for thisComponent in summary_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for summary_instructions
    summary_instructions.tStop = globalClock.getTime(format='float')
    summary_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('summary_instructions.stopped', summary_instructions.tStop)
    # check responses
    if advance_to_runs.keys in ['', [], None]:  # No response was made
        advance_to_runs.keys = None
    thisExp.addData('advance_to_runs.keys',advance_to_runs.keys)
    if advance_to_runs.keys != None:  # we had a response
        thisExp.addData('advance_to_runs.rt', advance_to_runs.rt)
        thisExp.addData('advance_to_runs.duration', advance_to_runs.duration)
    thisExp.nextEntry()
    # the Routine "summary_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    runs = data.TrialHandler2(
        name='runs',
        nReps=num_runs, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(runs)  # add the loop to the experiment
    thisRun = runs.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
    if thisRun != None:
        for paramName in thisRun:
            globals()[paramName] = thisRun[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisRun in runs:
        currentLoop = runs
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisRun.rgb)
        if thisRun != None:
            for paramName in thisRun:
                globals()[paramName] = thisRun[paramName]
        
        # --- Prepare to start Routine "get_ready" ---
        # create an object to store info about Routine get_ready
        get_ready = data.Routine(
            name='get_ready',
            components=[get_ready_text, advance_press],
        )
        get_ready.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for advance_press
        advance_press.keys = []
        advance_press.rt = []
        _advance_press_allKeys = []
        # Run 'Begin Routine' code from init_runs
        num_gold_coins = 0
        leftover_t = 0 #leftover time for fast RTs to run out at end of trial
        
        #pick coin side for the start of the run
        coin = random.randint(1, 2)
        if coin == 1:
            good_side = "left"
        else:
            good_side = "right"
          
        good_side = "right"
        #start at a new magic num for runs 2, 3
        if runs.thisN != 0 and num_rewarded != 0:
            num_switch = num_switch + 1
            print(num_switch)
        
        
        num_rewarded = 0 #start fresh for each run
            
        
        # store start times for get_ready
        get_ready.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        get_ready.tStart = globalClock.getTime(format='float')
        get_ready.status = STARTED
        thisExp.addData('get_ready.started', get_ready.tStart)
        get_ready.maxDuration = None
        # keep track of which components have finished
        get_readyComponents = get_ready.components
        for thisComponent in get_ready.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "get_ready" ---
        # if trial has changed, end Routine now
        if isinstance(runs, data.TrialHandler2) and thisRun.thisN != runs.thisTrial.thisN:
            continueRoutine = False
        get_ready.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *get_ready_text* updates
            
            # if get_ready_text is starting this frame...
            if get_ready_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                get_ready_text.frameNStart = frameN  # exact frame index
                get_ready_text.tStart = t  # local t and not account for scr refresh
                get_ready_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(get_ready_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'get_ready_text.started')
                # update status
                get_ready_text.status = STARTED
                get_ready_text.setAutoDraw(True)
            
            # if get_ready_text is active this frame...
            if get_ready_text.status == STARTED:
                # update params
                pass
            
            # *advance_press* updates
            waitOnFlip = False
            
            # if advance_press is starting this frame...
            if advance_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                advance_press.frameNStart = frameN  # exact frame index
                advance_press.tStart = t  # local t and not account for scr refresh
                advance_press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(advance_press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'advance_press.started')
                # update status
                advance_press.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(advance_press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(advance_press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if advance_press.status == STARTED and not waitOnFlip:
                theseKeys = advance_press.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _advance_press_allKeys.extend(theseKeys)
                if len(_advance_press_allKeys):
                    advance_press.keys = _advance_press_allKeys[-1].name  # just the last key pressed
                    advance_press.rt = _advance_press_allKeys[-1].rt
                    advance_press.duration = _advance_press_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                get_ready.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in get_ready.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "get_ready" ---
        for thisComponent in get_ready.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for get_ready
        get_ready.tStop = globalClock.getTime(format='float')
        get_ready.tStopRefresh = tThisFlipGlobal
        thisExp.addData('get_ready.stopped', get_ready.tStop)
        # check responses
        if advance_press.keys in ['', [], None]:  # No response was made
            advance_press.keys = None
        runs.addData('advance_press.keys',advance_press.keys)
        if advance_press.keys != None:  # we had a response
            runs.addData('advance_press.rt', advance_press.rt)
            runs.addData('advance_press.duration', advance_press.duration)
        # the Routine "get_ready" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "scanner_trigger" ---
        # create an object to store info about Routine scanner_trigger
        scanner_trigger = data.Routine(
            name='scanner_trigger',
            components=[scanner_text, key_resp],
        )
        scanner_trigger.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from trigger_code
        waitForScannerClock = core.Clock()
        fmriClock = core.Clock()
        # store start times for scanner_trigger
        scanner_trigger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        scanner_trigger.tStart = globalClock.getTime(format='float')
        scanner_trigger.status = STARTED
        thisExp.addData('scanner_trigger.started', scanner_trigger.tStart)
        scanner_trigger.maxDuration = None
        # keep track of which components have finished
        scanner_triggerComponents = scanner_trigger.components
        for thisComponent in scanner_trigger.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "scanner_trigger" ---
        # if trial has changed, end Routine now
        if isinstance(runs, data.TrialHandler2) and thisRun.thisN != runs.thisTrial.thisN:
            continueRoutine = False
        scanner_trigger.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *scanner_text* updates
            
            # if scanner_text is starting this frame...
            if scanner_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                scanner_text.frameNStart = frameN  # exact frame index
                scanner_text.tStart = t  # local t and not account for scr refresh
                scanner_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(scanner_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'scanner_text.started')
                # update status
                scanner_text.status = STARTED
                scanner_text.setAutoDraw(True)
            
            # if scanner_text is active this frame...
            if scanner_text.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['t'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                scanner_trigger.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in scanner_trigger.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "scanner_trigger" ---
        for thisComponent in scanner_trigger.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for scanner_trigger
        scanner_trigger.tStop = globalClock.getTime(format='float')
        scanner_trigger.tStopRefresh = tThisFlipGlobal
        thisExp.addData('scanner_trigger.stopped', scanner_trigger.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        runs.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            runs.addData('key_resp.rt', key_resp.rt)
            runs.addData('key_resp.duration', key_resp.duration)
        # the Routine "scanner_trigger" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=50.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "cue" ---
            # create an object to store info about Routine cue
            cue = data.Routine(
                name='cue',
                components=[left_box, right_box, cue_resp, all_keys],
            )
            cue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for cue_resp
            cue_resp.keys = []
            cue_resp.rt = []
            _cue_resp_allKeys = []
            # Run 'Begin Routine' code from good_side_code
            #if the participant has the right number of
            #correct guesses
            if num_rewarded == num_switch_list[num_switch]:
                if good_side == "right":
                    good_side = "left"
                elif good_side == "left":
                    good_side = "right"
                num_rewarded = 0
                num_switch = num_switch + 1 # we move one switch forward
                
                
            #if good_side has switched, make first correct choice
            #a gold coin
            
            
            
            # create starting attributes for all_keys
            all_keys.keys = []
            all_keys.rt = []
            _all_keys_allKeys = []
            # store start times for cue
            cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            cue.tStart = globalClock.getTime(format='float')
            cue.status = STARTED
            thisExp.addData('cue.started', cue.tStart)
            cue.maxDuration = None
            # keep track of which components have finished
            cueComponents = cue.components
            for thisComponent in cue.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "cue" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            cue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *left_box* updates
                
                # if left_box is starting this frame...
                if left_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    left_box.frameNStart = frameN  # exact frame index
                    left_box.tStart = t  # local t and not account for scr refresh
                    left_box.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(left_box, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_box.started')
                    # update status
                    left_box.status = STARTED
                    left_box.setAutoDraw(True)
                
                # if left_box is active this frame...
                if left_box.status == STARTED:
                    # update params
                    pass
                
                # if left_box is stopping this frame...
                if left_box.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > left_box.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        left_box.tStop = t  # not accounting for scr refresh
                        left_box.tStopRefresh = tThisFlipGlobal  # on global time
                        left_box.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'left_box.stopped')
                        # update status
                        left_box.status = FINISHED
                        left_box.setAutoDraw(False)
                
                # *right_box* updates
                
                # if right_box is starting this frame...
                if right_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    right_box.frameNStart = frameN  # exact frame index
                    right_box.tStart = t  # local t and not account for scr refresh
                    right_box.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(right_box, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_box.started')
                    # update status
                    right_box.status = STARTED
                    right_box.setAutoDraw(True)
                
                # if right_box is active this frame...
                if right_box.status == STARTED:
                    # update params
                    pass
                
                # if right_box is stopping this frame...
                if right_box.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > right_box.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        right_box.tStop = t  # not accounting for scr refresh
                        right_box.tStopRefresh = tThisFlipGlobal  # on global time
                        right_box.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'right_box.stopped')
                        # update status
                        right_box.status = FINISHED
                        right_box.setAutoDraw(False)
                
                # *cue_resp* updates
                waitOnFlip = False
                
                # if cue_resp is starting this frame...
                if cue_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_resp.frameNStart = frameN  # exact frame index
                    cue_resp.tStart = t  # local t and not account for scr refresh
                    cue_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_resp.started')
                    # update status
                    cue_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(cue_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(cue_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if cue_resp is stopping this frame...
                if cue_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_resp.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_resp.tStop = t  # not accounting for scr refresh
                        cue_resp.tStopRefresh = tThisFlipGlobal  # on global time
                        cue_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_resp.stopped')
                        # update status
                        cue_resp.status = FINISHED
                        cue_resp.status = FINISHED
                if cue_resp.status == STARTED and not waitOnFlip:
                    theseKeys = cue_resp.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _cue_resp_allKeys.extend(theseKeys)
                    if len(_cue_resp_allKeys):
                        cue_resp.keys = _cue_resp_allKeys[0].name  # just the first key pressed
                        cue_resp.rt = _cue_resp_allKeys[0].rt
                        cue_resp.duration = _cue_resp_allKeys[0].duration
                        # a response ends the routine
                        continueRoutine = False
                # Run 'Each Frame' code from good_side_code
                
                    
                
                # *all_keys* updates
                waitOnFlip = False
                
                # if all_keys is starting this frame...
                if all_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    all_keys.frameNStart = frameN  # exact frame index
                    all_keys.tStart = t  # local t and not account for scr refresh
                    all_keys.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(all_keys, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'all_keys.started')
                    # update status
                    all_keys.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(all_keys.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(all_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if all_keys is stopping this frame...
                if all_keys.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > all_keys.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        all_keys.tStop = t  # not accounting for scr refresh
                        all_keys.tStopRefresh = tThisFlipGlobal  # on global time
                        all_keys.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'all_keys.stopped')
                        # update status
                        all_keys.status = FINISHED
                        all_keys.status = FINISHED
                if all_keys.status == STARTED and not waitOnFlip:
                    theseKeys = all_keys.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                    _all_keys_allKeys.extend(theseKeys)
                    if len(_all_keys_allKeys):
                        all_keys.keys = [key.name for key in _all_keys_allKeys]  # storing all keys
                        all_keys.rt = [key.rt for key in _all_keys_allKeys]
                        all_keys.duration = [key.duration for key in _all_keys_allKeys]
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    cue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "cue" ---
            for thisComponent in cue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for cue
            cue.tStop = globalClock.getTime(format='float')
            cue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('cue.stopped', cue.tStop)
            # check responses
            if cue_resp.keys in ['', [], None]:  # No response was made
                cue_resp.keys = None
            trials.addData('cue_resp.keys',cue_resp.keys)
            if cue_resp.keys != None:  # we had a response
                trials.addData('cue_resp.rt', cue_resp.rt)
                trials.addData('cue_resp.duration', cue_resp.duration)
            # Run 'End Routine' code from good_side_code
            #if good_side has switched and they pick the good side 
            #for the first time, make first correct choice
            #a gold coin
            
            if(num_rewarded == 0 and sequence[exp_correct] == 0 and ((cue_resp.keys == '1' and good_side == "left") or (cue_resp.keys == '2' and good_side == "right"))):
                print("trying to switch")
                sequence[exp_correct] = 1
                #double check this logic:
                next_1 = sequence.index(1, exp_correct + 1)
                sequence[next_1] = 0
              
            #print("trialNum " + str(trials.thisN))
            #print("rewards" + str(num_rewarded))
            #print("prob seq" + str(sequence[exp_correct]))
            #print((cue_resp.keys == '1' and good_side == "left") or (cue_resp.keys == '2' and good_side == "right"))
            
            
            #Check if these need to be reward or just right
            # I'm guessing rewarded
            if(cue_resp.keys == '1' and good_side == "left"): 
                if(sequence[exp_correct] == 1):
                    num_rewarded = num_rewarded + 1
                    num_gold_coins = num_gold_coins + 1
                exp_correct = exp_correct + 1 
                
            
                
            if(cue_resp.keys == '2' and good_side == "right"):
                if(sequence[exp_correct] == 1):
                    num_rewarded = num_rewarded + 1
                    num_gold_coins = num_gold_coins + 1
                exp_correct = exp_correct + 1
                
                
            thisExp.addData('good_side', good_side)
            thisExp.addData('probability_sequence_value', sequence[exp_correct - 1])
            thisExp.addData('magic_number', num_switch_list[num_switch])
            thisExp.addData('num_rewarded', num_rewarded)
            thisExp.addData('num_switch', num_switch)
            thisExp.addData('total_correct', exp_correct)
            thisExp.addData('num_gold_coins', num_gold_coins)
            # check responses
            if all_keys.keys in ['', [], None]:  # No response was made
                all_keys.keys = None
            trials.addData('all_keys.keys',all_keys.keys)
            if all_keys.keys != None:  # we had a response
                trials.addData('all_keys.rt', all_keys.rt)
                trials.addData('all_keys.duration', all_keys.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if cue.maxDurationReached:
                routineTimer.addTime(-cue.maxDuration)
            elif cue.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "cue_response" ---
            # create an object to store info about Routine cue_response
            cue_response = data.Routine(
                name='cue_response',
                components=[left_box_response, right_box_response],
            )
            cue_response.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from cue_resp_feedback_code
            #initialize selection indicator
            selection_indicator = visual.Rect(
                win=win, name='polygon',
                width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
                ori=0.0, draggable=False, anchor='center',
                lineWidth=4.0,
                colorSpace='rgb', lineColor='white', fillColor=None,
                depth=-4.0, interpolate=True, 
                autoDraw = False)
                
                
            #initialize too slow message
            too_slow_text = visual.TextStim(win=win, name='too_slow_text',
            text='too slow',
            font='Arial',
            pos=(0, .1), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=1.0, 
            languageStyle='LTR',
            depth=-6.0, 
            autoDraw = False)
            
            #if response was made
            #get cue_resp.rt and figure out how much time is left over
            if(cue_resp.rt): 
                remaining_t = 1.5 - cue_resp.rt
                leftover_t = leftover_t + remaining_t
                #select right or left
                if cue_resp.keys == '1': 
                    position = (-0.3, -0.15)
                    selection_indicator.setPos(position)
                    selection_indicator.setAutoDraw(True)
                elif cue_resp.keys == '2':
                    position = (0.3, -0.15)
                    selection_indicator.setPos(position)
                    selection_indicator.setAutoDraw(True)
            
            #show too slow message    
            else:
                too_slow_text.setAutoDraw(True)
            # store start times for cue_response
            cue_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            cue_response.tStart = globalClock.getTime(format='float')
            cue_response.status = STARTED
            thisExp.addData('cue_response.started', cue_response.tStart)
            cue_response.maxDuration = None
            # keep track of which components have finished
            cue_responseComponents = cue_response.components
            for thisComponent in cue_response.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "cue_response" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            cue_response.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from cue_resp_feedback_code
                
                    
                
                
                # *left_box_response* updates
                
                # if left_box_response is starting this frame...
                if left_box_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    left_box_response.frameNStart = frameN  # exact frame index
                    left_box_response.tStart = t  # local t and not account for scr refresh
                    left_box_response.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(left_box_response, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_box_response.started')
                    # update status
                    left_box_response.status = STARTED
                    left_box_response.setAutoDraw(True)
                
                # if left_box_response is active this frame...
                if left_box_response.status == STARTED:
                    # update params
                    pass
                
                # if left_box_response is stopping this frame...
                if left_box_response.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > left_box_response.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        left_box_response.tStop = t  # not accounting for scr refresh
                        left_box_response.tStopRefresh = tThisFlipGlobal  # on global time
                        left_box_response.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'left_box_response.stopped')
                        # update status
                        left_box_response.status = FINISHED
                        left_box_response.setAutoDraw(False)
                
                # *right_box_response* updates
                
                # if right_box_response is starting this frame...
                if right_box_response.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    right_box_response.frameNStart = frameN  # exact frame index
                    right_box_response.tStart = t  # local t and not account for scr refresh
                    right_box_response.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(right_box_response, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_box_response.started')
                    # update status
                    right_box_response.status = STARTED
                    right_box_response.setAutoDraw(True)
                
                # if right_box_response is active this frame...
                if right_box_response.status == STARTED:
                    # update params
                    pass
                
                # if right_box_response is stopping this frame...
                if right_box_response.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > right_box_response.tStartRefresh + 1-frameTolerance:
                        # keep track of stop time/frame for later
                        right_box_response.tStop = t  # not accounting for scr refresh
                        right_box_response.tStopRefresh = tThisFlipGlobal  # on global time
                        right_box_response.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'right_box_response.stopped')
                        # update status
                        right_box_response.status = FINISHED
                        right_box_response.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    cue_response.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cue_response.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "cue_response" ---
            for thisComponent in cue_response.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for cue_response
            cue_response.tStop = globalClock.getTime(format='float')
            cue_response.tStopRefresh = tThisFlipGlobal
            thisExp.addData('cue_response.stopped', cue_response.tStop)
            # Run 'End Routine' code from cue_resp_feedback_code
            #turn off selection indicator
            selection_indicator.setAutoDraw(False)
            
            #turn off too slow
            too_slow_text.setAutoDraw(False)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if cue_response.maxDurationReached:
                routineTimer.addTime(-cue_response.maxDuration)
            elif cue_response.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "outcome" ---
            # create an object to store info about Routine outcome
            outcome = data.Routine(
                name='outcome',
                components=[outcome_left_box, outcome_right_box],
            )
            outcome.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from box_selection_code
            #if no press is made
            right_image = 'stimuli/box_transparent.png'
            left_image = 'stimuli/box_transparent.png'
            #if right press
            if cue_resp.keys=='1': 
                #look back one since exp_correct has been incremented
                if sequence[exp_correct - 1] == 1 and good_side == "left":
                    left_image = 'stimuli/box_coin_transparent.png'
                else:
                    left_image = 'stimuli/box_empty_transparent.png'
            #if left press
            # load in images of either coin box or empty box conditionally
            elif cue_resp.keys=='2': #same as above
                #look back one since exp_correct has been incremented
                if sequence[exp_correct -1 ] == 1 and good_side == "right":
                    right_image = 'stimuli/box_coin_transparent.png'
                else:
                    right_image = 'stimuli/box_empty_transparent.png'
            else:
                Choice="NA"
            outcome_left_box.setImage(left_image)
            outcome_right_box.setImage(right_image)
            # store start times for outcome
            outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            outcome.tStart = globalClock.getTime(format='float')
            outcome.status = STARTED
            thisExp.addData('outcome.started', outcome.tStart)
            outcome.maxDuration = None
            # keep track of which components have finished
            outcomeComponents = outcome.components
            for thisComponent in outcome.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "outcome" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            outcome.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *outcome_left_box* updates
                
                # if outcome_left_box is starting this frame...
                if outcome_left_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    outcome_left_box.frameNStart = frameN  # exact frame index
                    outcome_left_box.tStart = t  # local t and not account for scr refresh
                    outcome_left_box.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(outcome_left_box, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'outcome_left_box.started')
                    # update status
                    outcome_left_box.status = STARTED
                    outcome_left_box.setAutoDraw(True)
                
                # if outcome_left_box is active this frame...
                if outcome_left_box.status == STARTED:
                    # update params
                    pass
                
                # if outcome_left_box is stopping this frame...
                if outcome_left_box.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > outcome_left_box.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        outcome_left_box.tStop = t  # not accounting for scr refresh
                        outcome_left_box.tStopRefresh = tThisFlipGlobal  # on global time
                        outcome_left_box.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'outcome_left_box.stopped')
                        # update status
                        outcome_left_box.status = FINISHED
                        outcome_left_box.setAutoDraw(False)
                
                # *outcome_right_box* updates
                
                # if outcome_right_box is starting this frame...
                if outcome_right_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    outcome_right_box.frameNStart = frameN  # exact frame index
                    outcome_right_box.tStart = t  # local t and not account for scr refresh
                    outcome_right_box.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(outcome_right_box, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'outcome_right_box.started')
                    # update status
                    outcome_right_box.status = STARTED
                    outcome_right_box.setAutoDraw(True)
                
                # if outcome_right_box is active this frame...
                if outcome_right_box.status == STARTED:
                    # update params
                    pass
                
                # if outcome_right_box is stopping this frame...
                if outcome_right_box.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > outcome_right_box.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        outcome_right_box.tStop = t  # not accounting for scr refresh
                        outcome_right_box.tStopRefresh = tThisFlipGlobal  # on global time
                        outcome_right_box.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'outcome_right_box.stopped')
                        # update status
                        outcome_right_box.status = FINISHED
                        outcome_right_box.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    outcome.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in outcome.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "outcome" ---
            for thisComponent in outcome.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for outcome
            outcome.tStop = globalClock.getTime(format='float')
            outcome.tStopRefresh = tThisFlipGlobal
            thisExp.addData('outcome.stopped', outcome.tStop)
            # Run 'End Routine' code from box_selection_code
            if cue_resp.keys=='1':
                thisExp.addData('outcome_image', left_image)
            elif cue_resp.keys=='2':
                thisExp.addData('outcome_image', right_image)
            else:
                thisExp.addData('outcome_image', "no selection made")
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if outcome.maxDurationReached:
                routineTimer.addTime(-outcome.maxDuration)
            elif outcome.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "fixation" ---
            # create an object to store info about Routine fixation
            fixation = data.Routine(
                name='fixation',
                components=[fixation_cross],
            )
            fixation.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for fixation
            fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fixation.tStart = globalClock.getTime(format='float')
            fixation.status = STARTED
            thisExp.addData('fixation.started', fixation.tStart)
            fixation.maxDuration = None
            # keep track of which components have finished
            fixationComponents = fixation.components
            for thisComponent in fixation.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            fixation.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fixation_cross* updates
                
                # if fixation_cross is starting this frame...
                if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fixation_cross.frameNStart = frameN  # exact frame index
                    fixation_cross.tStart = t  # local t and not account for scr refresh
                    fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross.started')
                    # update status
                    fixation_cross.status = STARTED
                    fixation_cross.setAutoDraw(True)
                
                # if fixation_cross is active this frame...
                if fixation_cross.status == STARTED:
                    # update params
                    pass
                
                # if fixation_cross is stopping this frame...
                if fixation_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fixation_cross.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        fixation_cross.tStop = t  # not accounting for scr refresh
                        fixation_cross.tStopRefresh = tThisFlipGlobal  # on global time
                        fixation_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fixation_cross.stopped')
                        # update status
                        fixation_cross.status = FINISHED
                        fixation_cross.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fixation.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation" ---
            for thisComponent in fixation.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fixation
            fixation.tStop = globalClock.getTime(format='float')
            fixation.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fixation.stopped', fixation.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if fixation.maxDurationReached:
                routineTimer.addTime(-fixation.maxDuration)
            elif fixation.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.500000)
            thisExp.nextEntry()
            
        # completed 50.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # --- Prepare to start Routine "run_feedback" ---
        # create an object to store info about Routine run_feedback
        run_feedback = data.Routine(
            name='run_feedback',
            components=[coins_won_text],
        )
        run_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        coins_won_text.setText('Congratulations, you collected ' + str(num_gold_coins) + ' gold coins!!')
        # store start times for run_feedback
        run_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        run_feedback.tStart = globalClock.getTime(format='float')
        run_feedback.status = STARTED
        thisExp.addData('run_feedback.started', run_feedback.tStart)
        run_feedback.maxDuration = None
        # keep track of which components have finished
        run_feedbackComponents = run_feedback.components
        for thisComponent in run_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "run_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(runs, data.TrialHandler2) and thisRun.thisN != runs.thisTrial.thisN:
            continueRoutine = False
        run_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *coins_won_text* updates
            
            # if coins_won_text is starting this frame...
            if coins_won_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                coins_won_text.frameNStart = frameN  # exact frame index
                coins_won_text.tStart = t  # local t and not account for scr refresh
                coins_won_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(coins_won_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'coins_won_text.started')
                # update status
                coins_won_text.status = STARTED
                coins_won_text.setAutoDraw(True)
            
            # if coins_won_text is active this frame...
            if coins_won_text.status == STARTED:
                # update params
                pass
            
            # if coins_won_text is stopping this frame...
            if coins_won_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > coins_won_text.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    coins_won_text.tStop = t  # not accounting for scr refresh
                    coins_won_text.tStopRefresh = tThisFlipGlobal  # on global time
                    coins_won_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'coins_won_text.stopped')
                    # update status
                    coins_won_text.status = FINISHED
                    coins_won_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                run_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in run_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "run_feedback" ---
        for thisComponent in run_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for run_feedback
        run_feedback.tStop = globalClock.getTime(format='float')
        run_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('run_feedback.stopped', run_feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if run_feedback.maxDurationReached:
            routineTimer.addTime(-run_feedback.maxDuration)
        elif run_feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        
        # --- Prepare to start Routine "leftover_time_break" ---
        # create an object to store info about Routine leftover_time_break
        leftover_time_break = data.Routine(
            name='leftover_time_break',
            components=[fixation_end],
        )
        leftover_time_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        fixation_end.setText('Take a short break for  ' + str(leftover_t) + ' seconds!!')
        # store start times for leftover_time_break
        leftover_time_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        leftover_time_break.tStart = globalClock.getTime(format='float')
        leftover_time_break.status = STARTED
        thisExp.addData('leftover_time_break.started', leftover_time_break.tStart)
        leftover_time_break.maxDuration = None
        # keep track of which components have finished
        leftover_time_breakComponents = leftover_time_break.components
        for thisComponent in leftover_time_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "leftover_time_break" ---
        # if trial has changed, end Routine now
        if isinstance(runs, data.TrialHandler2) and thisRun.thisN != runs.thisTrial.thisN:
            continueRoutine = False
        leftover_time_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_end* updates
            
            # if fixation_end is starting this frame...
            if fixation_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_end.frameNStart = frameN  # exact frame index
                fixation_end.tStart = t  # local t and not account for scr refresh
                fixation_end.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_end, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_end.started')
                # update status
                fixation_end.status = STARTED
                fixation_end.setAutoDraw(True)
            
            # if fixation_end is active this frame...
            if fixation_end.status == STARTED:
                # update params
                pass
            
            # if fixation_end is stopping this frame...
            if fixation_end.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_end.tStartRefresh + leftover_t-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_end.tStop = t  # not accounting for scr refresh
                    fixation_end.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_end.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_end.stopped')
                    # update status
                    fixation_end.status = FINISHED
                    fixation_end.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                leftover_time_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in leftover_time_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "leftover_time_break" ---
        for thisComponent in leftover_time_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for leftover_time_break
        leftover_time_break.tStop = globalClock.getTime(format='float')
        leftover_time_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('leftover_time_break.stopped', leftover_time_break.tStop)
        # the Routine "leftover_time_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed num_runs repeats of 'runs'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_exp" ---
    # create an object to store info about Routine end_exp
    end_exp = data.Routine(
        name='end_exp',
        components=[],
    )
    end_exp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end_exp
    end_exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_exp.tStart = globalClock.getTime(format='float')
    end_exp.status = STARTED
    thisExp.addData('end_exp.started', end_exp.tStart)
    end_exp.maxDuration = None
    # keep track of which components have finished
    end_expComponents = end_exp.components
    for thisComponent in end_exp.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_exp" ---
    end_exp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end_exp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_exp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_exp" ---
    for thisComponent in end_exp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_exp
    end_exp.tStop = globalClock.getTime(format='float')
    end_exp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_exp.stopped', end_exp.tStop)
    thisExp.nextEntry()
    # the Routine "end_exp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
