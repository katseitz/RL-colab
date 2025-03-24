#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Sun Mar 23 19:23:13 2025
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
# Run 'Before Experiment' code from iti_code
import pandas as pd
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
    'handedness': ["right", "left"],
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
        originPath='/Users/katharinaseitz/Documents/projects/RL-colab/practice_reinforcement_learning_reversal_lastrun.py',
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
            monitor='kats_mac', color=[-1,-1,-1], colorSpace='rgb',
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
    if deviceManager.getDevice('advance_press_pd') is None:
        # initialise advance_press_pd
        advance_press_pd = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='advance_press_pd',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('find_box_press') is None:
        # initialise find_box_press
        find_box_press = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='find_box_press',
        )
    if deviceManager.getDevice('cue_resp_switch') is None:
        # initialise cue_resp_switch
        cue_resp_switch = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='cue_resp_switch',
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
    
    # --- Initialize components for Routine "prac_dir" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='GAME OF COINS\n\nIn this game, your job is collect as many gold coins as possible. \n\n\nPress the space bar to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    advance_press_pd = keyboard.Keyboard(deviceName='advance_press_pd')
    
    # --- Initialize components for Routine "first_box" ---
    show_stimuli = visual.ImageStim(
        win=win,
        name='show_stimuli', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    stimuli_text = visual.TextStim(win=win, name='stimuli_text',
        text='The coins are hidden in magical boxes, like this one:\n\n\n\n\n\n\n\nCan you open the box to see if it has a coin? Press the "1" key to open the box.',
        font='Arial',
        pos=(0, .1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "first_coin" ---
    first_coin_img = visual.ImageStim(
        win=win,
        name='first_coin_img', 
        image='stimuli/box_coin_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    magic_coin_text = visual.TextStim(win=win, name='magic_coin_text',
        text='This box is magical! It contains a coin.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "prac_both_instructions" ---
    prac_both_text = visual.TextStim(win=win, name='prac_both_text',
        text='In the game, there will be two boxes, but only one is magical. Try finding the magic box. The "1" key opens the left box and the "2" key opens the right box. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    prac_left_3 = visual.ImageStim(
        win=win,
        name='prac_left_3', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    prac_right_3 = visual.ImageStim(
        win=win,
        name='prac_right_3', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "prac_both" ---
    prac_left_2 = visual.ImageStim(
        win=win,
        name='prac_left_2', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac_right_2 = visual.ImageStim(
        win=win,
        name='prac_right_2', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    find_box_press = keyboard.Keyboard(deviceName='find_box_press')
    
    # --- Initialize components for Routine "prac_both_selection" ---
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
    
    # --- Initialize components for Routine "prac_both_outcome" ---
    prac_left_box = visual.ImageStim(
        win=win,
        name='prac_left_box', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    prac_right_box = visual.ImageStim(
        win=win,
        name='prac_right_box', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "switch_instruc" ---
    switch_instructions = visual.TextStim(win=win, name='switch_instructions',
        text='Sometimes, the magical box switches sides\n\nTry finding the magical box again!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "switch_cue" ---
    left_box_switch = visual.ImageStim(
        win=win,
        name='left_box_switch', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_box_switch = visual.ImageStim(
        win=win,
        name='right_box_switch', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    cue_resp_switch = keyboard.Keyboard(deviceName='cue_resp_switch')
    
    # --- Initialize components for Routine "switch_select" ---
    left_switch_selec = visual.ImageStim(
        win=win,
        name='left_switch_selec', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_switch_selec = visual.ImageStim(
        win=win,
        name='right_switch_selec', 
        image='stimuli/box_transparent.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "switch_out" ---
    left_box_switch_out = visual.ImageStim(
        win=win,
        name='left_box_switch_out', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    right_box_switch_out = visual.ImageStim(
        win=win,
        name='right_box_switch_out', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.3, -0.1), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "no_coin_instruc" ---
    summary_text = visual.TextStim(win=win, name='summary_text',
        text='Sometimes, even the magical box does not have a coin. This makes it harder to find.\n\n\nCan you still find the magic box?\n\nPress the space bar to continue.',
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
    # Run 'Begin Experiment' code from iti_code
    #read in ISI and ITI jitters
    jitters = pd.read_csv('rl_reversal_jitters.csv')
    
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
    
    # --- Prepare to start Routine "prac_dir" ---
    # create an object to store info about Routine prac_dir
    prac_dir = data.Routine(
        name='prac_dir',
        components=[welcome_text, advance_press_pd],
    )
    prac_dir.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for advance_press_pd
    advance_press_pd.keys = []
    advance_press_pd.rt = []
    _advance_press_pd_allKeys = []
    # store start times for prac_dir
    prac_dir.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prac_dir.tStart = globalClock.getTime(format='float')
    prac_dir.status = STARTED
    thisExp.addData('prac_dir.started', prac_dir.tStart)
    prac_dir.maxDuration = None
    # keep track of which components have finished
    prac_dirComponents = prac_dir.components
    for thisComponent in prac_dir.components:
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
    
    # --- Run Routine "prac_dir" ---
    prac_dir.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *advance_press_pd* updates
        waitOnFlip = False
        
        # if advance_press_pd is starting this frame...
        if advance_press_pd.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            advance_press_pd.frameNStart = frameN  # exact frame index
            advance_press_pd.tStart = t  # local t and not account for scr refresh
            advance_press_pd.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(advance_press_pd, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'advance_press_pd.started')
            # update status
            advance_press_pd.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(advance_press_pd.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(advance_press_pd.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if advance_press_pd.status == STARTED and not waitOnFlip:
            theseKeys = advance_press_pd.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _advance_press_pd_allKeys.extend(theseKeys)
            if len(_advance_press_pd_allKeys):
                advance_press_pd.keys = _advance_press_pd_allKeys[-1].name  # just the last key pressed
                advance_press_pd.rt = _advance_press_pd_allKeys[-1].rt
                advance_press_pd.duration = _advance_press_pd_allKeys[-1].duration
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
            prac_dir.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_dir.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_dir" ---
    for thisComponent in prac_dir.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prac_dir
    prac_dir.tStop = globalClock.getTime(format='float')
    prac_dir.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prac_dir.stopped', prac_dir.tStop)
    # check responses
    if advance_press_pd.keys in ['', [], None]:  # No response was made
        advance_press_pd.keys = None
    thisExp.addData('advance_press_pd.keys',advance_press_pd.keys)
    if advance_press_pd.keys != None:  # we had a response
        thisExp.addData('advance_press_pd.rt', advance_press_pd.rt)
        thisExp.addData('advance_press_pd.duration', advance_press_pd.duration)
    thisExp.nextEntry()
    # the Routine "prac_dir" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "first_box" ---
    # create an object to store info about Routine first_box
    first_box = data.Routine(
        name='first_box',
        components=[show_stimuli, stimuli_text, key_resp],
    )
    first_box.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for first_box
    first_box.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    first_box.tStart = globalClock.getTime(format='float')
    first_box.status = STARTED
    thisExp.addData('first_box.started', first_box.tStart)
    first_box.maxDuration = None
    # keep track of which components have finished
    first_boxComponents = first_box.components
    for thisComponent in first_box.components:
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
    
    # --- Run Routine "first_box" ---
    first_box.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *show_stimuli* updates
        
        # if show_stimuli is starting this frame...
        if show_stimuli.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            show_stimuli.frameNStart = frameN  # exact frame index
            show_stimuli.tStart = t  # local t and not account for scr refresh
            show_stimuli.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(show_stimuli, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'show_stimuli.started')
            # update status
            show_stimuli.status = STARTED
            show_stimuli.setAutoDraw(True)
        
        # if show_stimuli is active this frame...
        if show_stimuli.status == STARTED:
            # update params
            pass
        
        # *stimuli_text* updates
        
        # if stimuli_text is starting this frame...
        if stimuli_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            stimuli_text.frameNStart = frameN  # exact frame index
            stimuli_text.tStart = t  # local t and not account for scr refresh
            stimuli_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stimuli_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'stimuli_text.started')
            # update status
            stimuli_text.status = STARTED
            stimuli_text.setAutoDraw(True)
        
        # if stimuli_text is active this frame...
        if stimuli_text.status == STARTED:
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
            theseKeys = key_resp.getKeys(keyList=['1'], ignoreKeys=["escape"], waitRelease=False)
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
            first_box.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in first_box.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "first_box" ---
    for thisComponent in first_box.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for first_box
    first_box.tStop = globalClock.getTime(format='float')
    first_box.tStopRefresh = tThisFlipGlobal
    thisExp.addData('first_box.stopped', first_box.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "first_box" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "first_coin" ---
    # create an object to store info about Routine first_coin
    first_coin = data.Routine(
        name='first_coin',
        components=[first_coin_img, magic_coin_text],
    )
    first_coin.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for first_coin
    first_coin.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    first_coin.tStart = globalClock.getTime(format='float')
    first_coin.status = STARTED
    thisExp.addData('first_coin.started', first_coin.tStart)
    first_coin.maxDuration = None
    # keep track of which components have finished
    first_coinComponents = first_coin.components
    for thisComponent in first_coin.components:
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
    
    # --- Run Routine "first_coin" ---
    first_coin.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *first_coin_img* updates
        
        # if first_coin_img is starting this frame...
        if first_coin_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            first_coin_img.frameNStart = frameN  # exact frame index
            first_coin_img.tStart = t  # local t and not account for scr refresh
            first_coin_img.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(first_coin_img, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'first_coin_img.started')
            # update status
            first_coin_img.status = STARTED
            first_coin_img.setAutoDraw(True)
        
        # if first_coin_img is active this frame...
        if first_coin_img.status == STARTED:
            # update params
            pass
        
        # if first_coin_img is stopping this frame...
        if first_coin_img.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > first_coin_img.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                first_coin_img.tStop = t  # not accounting for scr refresh
                first_coin_img.tStopRefresh = tThisFlipGlobal  # on global time
                first_coin_img.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'first_coin_img.stopped')
                # update status
                first_coin_img.status = FINISHED
                first_coin_img.setAutoDraw(False)
        
        # *magic_coin_text* updates
        
        # if magic_coin_text is starting this frame...
        if magic_coin_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            magic_coin_text.frameNStart = frameN  # exact frame index
            magic_coin_text.tStart = t  # local t and not account for scr refresh
            magic_coin_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(magic_coin_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'magic_coin_text.started')
            # update status
            magic_coin_text.status = STARTED
            magic_coin_text.setAutoDraw(True)
        
        # if magic_coin_text is active this frame...
        if magic_coin_text.status == STARTED:
            # update params
            pass
        
        # if magic_coin_text is stopping this frame...
        if magic_coin_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > magic_coin_text.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                magic_coin_text.tStop = t  # not accounting for scr refresh
                magic_coin_text.tStopRefresh = tThisFlipGlobal  # on global time
                magic_coin_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'magic_coin_text.stopped')
                # update status
                magic_coin_text.status = FINISHED
                magic_coin_text.setAutoDraw(False)
        
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
            first_coin.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in first_coin.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "first_coin" ---
    for thisComponent in first_coin.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for first_coin
    first_coin.tStop = globalClock.getTime(format='float')
    first_coin.tStopRefresh = tThisFlipGlobal
    thisExp.addData('first_coin.stopped', first_coin.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if first_coin.maxDurationReached:
        routineTimer.addTime(-first_coin.maxDuration)
    elif first_coin.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.500000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "prac_both_instructions" ---
    # create an object to store info about Routine prac_both_instructions
    prac_both_instructions = data.Routine(
        name='prac_both_instructions',
        components=[prac_both_text, prac_left_3, prac_right_3],
    )
    prac_both_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for prac_both_instructions
    prac_both_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    prac_both_instructions.tStart = globalClock.getTime(format='float')
    prac_both_instructions.status = STARTED
    thisExp.addData('prac_both_instructions.started', prac_both_instructions.tStart)
    prac_both_instructions.maxDuration = None
    # keep track of which components have finished
    prac_both_instructionsComponents = prac_both_instructions.components
    for thisComponent in prac_both_instructions.components:
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
    
    # --- Run Routine "prac_both_instructions" ---
    prac_both_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *prac_both_text* updates
        
        # if prac_both_text is starting this frame...
        if prac_both_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_both_text.frameNStart = frameN  # exact frame index
            prac_both_text.tStart = t  # local t and not account for scr refresh
            prac_both_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_both_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_both_text.started')
            # update status
            prac_both_text.status = STARTED
            prac_both_text.setAutoDraw(True)
        
        # if prac_both_text is active this frame...
        if prac_both_text.status == STARTED:
            # update params
            pass
        
        # if prac_both_text is stopping this frame...
        if prac_both_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prac_both_text.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                prac_both_text.tStop = t  # not accounting for scr refresh
                prac_both_text.tStopRefresh = tThisFlipGlobal  # on global time
                prac_both_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_both_text.stopped')
                # update status
                prac_both_text.status = FINISHED
                prac_both_text.setAutoDraw(False)
        
        # *prac_left_3* updates
        
        # if prac_left_3 is starting this frame...
        if prac_left_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_left_3.frameNStart = frameN  # exact frame index
            prac_left_3.tStart = t  # local t and not account for scr refresh
            prac_left_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_left_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_left_3.started')
            # update status
            prac_left_3.status = STARTED
            prac_left_3.setAutoDraw(True)
        
        # if prac_left_3 is active this frame...
        if prac_left_3.status == STARTED:
            # update params
            pass
        
        # if prac_left_3 is stopping this frame...
        if prac_left_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prac_left_3.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                prac_left_3.tStop = t  # not accounting for scr refresh
                prac_left_3.tStopRefresh = tThisFlipGlobal  # on global time
                prac_left_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_left_3.stopped')
                # update status
                prac_left_3.status = FINISHED
                prac_left_3.setAutoDraw(False)
        
        # *prac_right_3* updates
        
        # if prac_right_3 is starting this frame...
        if prac_right_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            prac_right_3.frameNStart = frameN  # exact frame index
            prac_right_3.tStart = t  # local t and not account for scr refresh
            prac_right_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(prac_right_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'prac_right_3.started')
            # update status
            prac_right_3.status = STARTED
            prac_right_3.setAutoDraw(True)
        
        # if prac_right_3 is active this frame...
        if prac_right_3.status == STARTED:
            # update params
            pass
        
        # if prac_right_3 is stopping this frame...
        if prac_right_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > prac_right_3.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                prac_right_3.tStop = t  # not accounting for scr refresh
                prac_right_3.tStopRefresh = tThisFlipGlobal  # on global time
                prac_right_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_right_3.stopped')
                # update status
                prac_right_3.status = FINISHED
                prac_right_3.setAutoDraw(False)
        
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
            prac_both_instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in prac_both_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "prac_both_instructions" ---
    for thisComponent in prac_both_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for prac_both_instructions
    prac_both_instructions.tStop = globalClock.getTime(format='float')
    prac_both_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('prac_both_instructions.stopped', prac_both_instructions.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if prac_both_instructions.maxDurationReached:
        routineTimer.addTime(-prac_both_instructions.maxDuration)
    elif prac_both_instructions.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.500000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    find_magic_box_loop = data.TrialHandler2(
        name='find_magic_box_loop',
        nReps=7.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(find_magic_box_loop)  # add the loop to the experiment
    thisFind_magic_box_loop = find_magic_box_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFind_magic_box_loop.rgb)
    if thisFind_magic_box_loop != None:
        for paramName in thisFind_magic_box_loop:
            globals()[paramName] = thisFind_magic_box_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFind_magic_box_loop in find_magic_box_loop:
        currentLoop = find_magic_box_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFind_magic_box_loop.rgb)
        if thisFind_magic_box_loop != None:
            for paramName in thisFind_magic_box_loop:
                globals()[paramName] = thisFind_magic_box_loop[paramName]
        
        # --- Prepare to start Routine "prac_both" ---
        # create an object to store info about Routine prac_both
        prac_both = data.Routine(
            name='prac_both',
            components=[prac_left_2, prac_right_2, find_box_press],
        )
        prac_both.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for find_box_press
        find_box_press.keys = []
        find_box_press.rt = []
        _find_box_press_allKeys = []
        # store start times for prac_both
        prac_both.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_both.tStart = globalClock.getTime(format='float')
        prac_both.status = STARTED
        thisExp.addData('prac_both.started', prac_both.tStart)
        prac_both.maxDuration = None
        # keep track of which components have finished
        prac_bothComponents = prac_both.components
        for thisComponent in prac_both.components:
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
        
        # --- Run Routine "prac_both" ---
        # if trial has changed, end Routine now
        if isinstance(find_magic_box_loop, data.TrialHandler2) and thisFind_magic_box_loop.thisN != find_magic_box_loop.thisTrial.thisN:
            continueRoutine = False
        prac_both.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_left_2* updates
            
            # if prac_left_2 is starting this frame...
            if prac_left_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_left_2.frameNStart = frameN  # exact frame index
                prac_left_2.tStart = t  # local t and not account for scr refresh
                prac_left_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_left_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_left_2.started')
                # update status
                prac_left_2.status = STARTED
                prac_left_2.setAutoDraw(True)
            
            # if prac_left_2 is active this frame...
            if prac_left_2.status == STARTED:
                # update params
                pass
            
            # *prac_right_2* updates
            
            # if prac_right_2 is starting this frame...
            if prac_right_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_right_2.frameNStart = frameN  # exact frame index
                prac_right_2.tStart = t  # local t and not account for scr refresh
                prac_right_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_right_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_right_2.started')
                # update status
                prac_right_2.status = STARTED
                prac_right_2.setAutoDraw(True)
            
            # if prac_right_2 is active this frame...
            if prac_right_2.status == STARTED:
                # update params
                pass
            
            # *find_box_press* updates
            waitOnFlip = False
            
            # if find_box_press is starting this frame...
            if find_box_press.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                find_box_press.frameNStart = frameN  # exact frame index
                find_box_press.tStart = t  # local t and not account for scr refresh
                find_box_press.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(find_box_press, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'find_box_press.started')
                # update status
                find_box_press.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(find_box_press.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(find_box_press.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if find_box_press.status == STARTED and not waitOnFlip:
                theseKeys = find_box_press.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _find_box_press_allKeys.extend(theseKeys)
                if len(_find_box_press_allKeys):
                    find_box_press.keys = _find_box_press_allKeys[-1].name  # just the last key pressed
                    find_box_press.rt = _find_box_press_allKeys[-1].rt
                    find_box_press.duration = _find_box_press_allKeys[-1].duration
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
                prac_both.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_both.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_both" ---
        for thisComponent in prac_both.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_both
        prac_both.tStop = globalClock.getTime(format='float')
        prac_both.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_both.stopped', prac_both.tStop)
        # check responses
        if find_box_press.keys in ['', [], None]:  # No response was made
            find_box_press.keys = None
        find_magic_box_loop.addData('find_box_press.keys',find_box_press.keys)
        if find_box_press.keys != None:  # we had a response
            find_magic_box_loop.addData('find_box_press.rt', find_box_press.rt)
            find_magic_box_loop.addData('find_box_press.duration', find_box_press.duration)
        # the Routine "prac_both" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "prac_both_selection" ---
        # create an object to store info about Routine prac_both_selection
        prac_both_selection = data.Routine(
            name='prac_both_selection',
            components=[left_box_lr, right_box_lr],
        )
        prac_both_selection.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from prac_left_feedback
        #initialize selection indicator
        if(find_box_press.keys == '1'):
            position = (-0.3, -0.15)
        else:
           position = (0.3, -0.15) 
        
        
        pr_selection_indicator = visual.Rect(
            win=win, name='polygon',
            width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
            ori=0.0, draggable=False, anchor='center',
            lineWidth=4.0,
            pos = position,
            colorSpace='rgb', lineColor='white', fillColor=None,
            depth=-4.0, interpolate=True, 
            autoDraw = True)
         
        
        # Run 'Begin Routine' code from prac_box_code_2
        #if no press is made
        right_image = 'stimuli/box_transparent.png'
        left_image = 'stimuli/box_transparent.png'
        
        #left box is magical
        if find_box_press.keys=='1': 
            left_image = 'stimuli/box_coin_transparent.png'
        else:
            right_image = 'stimuli/box_empty_transparent.png'
        
        # store start times for prac_both_selection
        prac_both_selection.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_both_selection.tStart = globalClock.getTime(format='float')
        prac_both_selection.status = STARTED
        thisExp.addData('prac_both_selection.started', prac_both_selection.tStart)
        prac_both_selection.maxDuration = None
        # keep track of which components have finished
        prac_both_selectionComponents = prac_both_selection.components
        for thisComponent in prac_both_selection.components:
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
        
        # --- Run Routine "prac_both_selection" ---
        # if trial has changed, end Routine now
        if isinstance(find_magic_box_loop, data.TrialHandler2) and thisFind_magic_box_loop.thisN != find_magic_box_loop.thisTrial.thisN:
            continueRoutine = False
        prac_both_selection.forceEnded = routineForceEnded = not continueRoutine
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
                prac_both_selection.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_both_selection.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_both_selection" ---
        for thisComponent in prac_both_selection.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_both_selection
        prac_both_selection.tStop = globalClock.getTime(format='float')
        prac_both_selection.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_both_selection.stopped', prac_both_selection.tStop)
        # Run 'End Routine' code from prac_left_feedback
        #turn off selection indicator
        pr_selection_indicator.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_both_selection.maxDurationReached:
            routineTimer.addTime(-prac_both_selection.maxDuration)
        elif prac_both_selection.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "prac_both_outcome" ---
        # create an object to store info about Routine prac_both_outcome
        prac_both_outcome = data.Routine(
            name='prac_both_outcome',
            components=[prac_left_box, prac_right_box],
        )
        prac_both_outcome.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        prac_left_box.setImage(left_image)
        prac_right_box.setImage(right_image)
        # store start times for prac_both_outcome
        prac_both_outcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        prac_both_outcome.tStart = globalClock.getTime(format='float')
        prac_both_outcome.status = STARTED
        thisExp.addData('prac_both_outcome.started', prac_both_outcome.tStart)
        prac_both_outcome.maxDuration = None
        # keep track of which components have finished
        prac_both_outcomeComponents = prac_both_outcome.components
        for thisComponent in prac_both_outcome.components:
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
        
        # --- Run Routine "prac_both_outcome" ---
        # if trial has changed, end Routine now
        if isinstance(find_magic_box_loop, data.TrialHandler2) and thisFind_magic_box_loop.thisN != find_magic_box_loop.thisTrial.thisN:
            continueRoutine = False
        prac_both_outcome.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *prac_left_box* updates
            
            # if prac_left_box is starting this frame...
            if prac_left_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_left_box.frameNStart = frameN  # exact frame index
                prac_left_box.tStart = t  # local t and not account for scr refresh
                prac_left_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_left_box, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_left_box.started')
                # update status
                prac_left_box.status = STARTED
                prac_left_box.setAutoDraw(True)
            
            # if prac_left_box is active this frame...
            if prac_left_box.status == STARTED:
                # update params
                pass
            
            # if prac_left_box is stopping this frame...
            if prac_left_box.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_left_box.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_left_box.tStop = t  # not accounting for scr refresh
                    prac_left_box.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_left_box.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_left_box.stopped')
                    # update status
                    prac_left_box.status = FINISHED
                    prac_left_box.setAutoDraw(False)
            
            # *prac_right_box* updates
            
            # if prac_right_box is starting this frame...
            if prac_right_box.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                prac_right_box.frameNStart = frameN  # exact frame index
                prac_right_box.tStart = t  # local t and not account for scr refresh
                prac_right_box.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prac_right_box, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prac_right_box.started')
                # update status
                prac_right_box.status = STARTED
                prac_right_box.setAutoDraw(True)
            
            # if prac_right_box is active this frame...
            if prac_right_box.status == STARTED:
                # update params
                pass
            
            # if prac_right_box is stopping this frame...
            if prac_right_box.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > prac_right_box.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    prac_right_box.tStop = t  # not accounting for scr refresh
                    prac_right_box.tStopRefresh = tThisFlipGlobal  # on global time
                    prac_right_box.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prac_right_box.stopped')
                    # update status
                    prac_right_box.status = FINISHED
                    prac_right_box.setAutoDraw(False)
            
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
                prac_both_outcome.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in prac_both_outcome.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "prac_both_outcome" ---
        for thisComponent in prac_both_outcome.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for prac_both_outcome
        prac_both_outcome.tStop = globalClock.getTime(format='float')
        prac_both_outcome.tStopRefresh = tThisFlipGlobal
        thisExp.addData('prac_both_outcome.stopped', prac_both_outcome.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if prac_both_outcome.maxDurationReached:
            routineTimer.addTime(-prac_both_outcome.maxDuration)
        elif prac_both_outcome.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 7.0 repeats of 'find_magic_box_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "switch_instruc" ---
    # create an object to store info about Routine switch_instruc
    switch_instruc = data.Routine(
        name='switch_instruc',
        components=[switch_instructions],
    )
    switch_instruc.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for switch_instruc
    switch_instruc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    switch_instruc.tStart = globalClock.getTime(format='float')
    switch_instruc.status = STARTED
    thisExp.addData('switch_instruc.started', switch_instruc.tStart)
    switch_instruc.maxDuration = None
    # keep track of which components have finished
    switch_instrucComponents = switch_instruc.components
    for thisComponent in switch_instruc.components:
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
    
    # --- Run Routine "switch_instruc" ---
    switch_instruc.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *switch_instructions* updates
        
        # if switch_instructions is starting this frame...
        if switch_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            switch_instructions.frameNStart = frameN  # exact frame index
            switch_instructions.tStart = t  # local t and not account for scr refresh
            switch_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(switch_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'switch_instructions.started')
            # update status
            switch_instructions.status = STARTED
            switch_instructions.setAutoDraw(True)
        
        # if switch_instructions is active this frame...
        if switch_instructions.status == STARTED:
            # update params
            pass
        
        # if switch_instructions is stopping this frame...
        if switch_instructions.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > switch_instructions.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                switch_instructions.tStop = t  # not accounting for scr refresh
                switch_instructions.tStopRefresh = tThisFlipGlobal  # on global time
                switch_instructions.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'switch_instructions.stopped')
                # update status
                switch_instructions.status = FINISHED
                switch_instructions.setAutoDraw(False)
        
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
            switch_instruc.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in switch_instruc.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "switch_instruc" ---
    for thisComponent in switch_instruc.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for switch_instruc
    switch_instruc.tStop = globalClock.getTime(format='float')
    switch_instruc.tStopRefresh = tThisFlipGlobal
    thisExp.addData('switch_instruc.stopped', switch_instruc.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if switch_instruc.maxDurationReached:
        routineTimer.addTime(-switch_instruc.maxDuration)
    elif switch_instruc.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    switch_side_loop = data.TrialHandler2(
        name='switch_side_loop',
        nReps=10.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(switch_side_loop)  # add the loop to the experiment
    thisSwitch_side_loop = switch_side_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSwitch_side_loop.rgb)
    if thisSwitch_side_loop != None:
        for paramName in thisSwitch_side_loop:
            globals()[paramName] = thisSwitch_side_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSwitch_side_loop in switch_side_loop:
        currentLoop = switch_side_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSwitch_side_loop.rgb)
        if thisSwitch_side_loop != None:
            for paramName in thisSwitch_side_loop:
                globals()[paramName] = thisSwitch_side_loop[paramName]
        
        # --- Prepare to start Routine "switch_cue" ---
        # create an object to store info about Routine switch_cue
        switch_cue = data.Routine(
            name='switch_cue',
            components=[left_box_switch, right_box_switch, cue_resp_switch],
        )
        switch_cue.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for cue_resp_switch
        cue_resp_switch.keys = []
        cue_resp_switch.rt = []
        _cue_resp_switch_allKeys = []
        # store start times for switch_cue
        switch_cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        switch_cue.tStart = globalClock.getTime(format='float')
        switch_cue.status = STARTED
        thisExp.addData('switch_cue.started', switch_cue.tStart)
        switch_cue.maxDuration = None
        # keep track of which components have finished
        switch_cueComponents = switch_cue.components
        for thisComponent in switch_cue.components:
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
        
        # --- Run Routine "switch_cue" ---
        # if trial has changed, end Routine now
        if isinstance(switch_side_loop, data.TrialHandler2) and thisSwitch_side_loop.thisN != switch_side_loop.thisTrial.thisN:
            continueRoutine = False
        switch_cue.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *left_box_switch* updates
            
            # if left_box_switch is starting this frame...
            if left_box_switch.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_box_switch.frameNStart = frameN  # exact frame index
                left_box_switch.tStart = t  # local t and not account for scr refresh
                left_box_switch.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_box_switch, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_box_switch.started')
                # update status
                left_box_switch.status = STARTED
                left_box_switch.setAutoDraw(True)
            
            # if left_box_switch is active this frame...
            if left_box_switch.status == STARTED:
                # update params
                pass
            
            # *right_box_switch* updates
            
            # if right_box_switch is starting this frame...
            if right_box_switch.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_box_switch.frameNStart = frameN  # exact frame index
                right_box_switch.tStart = t  # local t and not account for scr refresh
                right_box_switch.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_box_switch, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_box_switch.started')
                # update status
                right_box_switch.status = STARTED
                right_box_switch.setAutoDraw(True)
            
            # if right_box_switch is active this frame...
            if right_box_switch.status == STARTED:
                # update params
                pass
            
            # *cue_resp_switch* updates
            waitOnFlip = False
            
            # if cue_resp_switch is starting this frame...
            if cue_resp_switch.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_resp_switch.frameNStart = frameN  # exact frame index
                cue_resp_switch.tStart = t  # local t and not account for scr refresh
                cue_resp_switch.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_resp_switch, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_resp_switch.started')
                # update status
                cue_resp_switch.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(cue_resp_switch.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(cue_resp_switch.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if cue_resp_switch.status == STARTED and not waitOnFlip:
                theseKeys = cue_resp_switch.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                _cue_resp_switch_allKeys.extend(theseKeys)
                if len(_cue_resp_switch_allKeys):
                    cue_resp_switch.keys = _cue_resp_switch_allKeys[0].name  # just the first key pressed
                    cue_resp_switch.rt = _cue_resp_switch_allKeys[0].rt
                    cue_resp_switch.duration = _cue_resp_switch_allKeys[0].duration
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
                switch_cue.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in switch_cue.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "switch_cue" ---
        for thisComponent in switch_cue.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for switch_cue
        switch_cue.tStop = globalClock.getTime(format='float')
        switch_cue.tStopRefresh = tThisFlipGlobal
        thisExp.addData('switch_cue.stopped', switch_cue.tStop)
        # check responses
        if cue_resp_switch.keys in ['', [], None]:  # No response was made
            cue_resp_switch.keys = None
        switch_side_loop.addData('cue_resp_switch.keys',cue_resp_switch.keys)
        if cue_resp_switch.keys != None:  # we had a response
            switch_side_loop.addData('cue_resp_switch.rt', cue_resp_switch.rt)
            switch_side_loop.addData('cue_resp_switch.duration', cue_resp_switch.duration)
        # the Routine "switch_cue" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "switch_select" ---
        # create an object to store info about Routine switch_select
        switch_select = data.Routine(
            name='switch_select',
            components=[left_switch_selec, right_switch_selec],
        )
        switch_select.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from prac_left_feedback_2
        #initialize selection indicator
        if(cue_resp_switch.keys == '1'):
            position = (-0.3, -0.15)
        else:
           position = (0.3, -0.15) 
        
        
        pr_selection_indicator = visual.Rect(
            win=win, name='polygon',
            width=(0.5, 0.5)[0], height=(0.5, 0.5)[1],
            ori=0.0, draggable=False, anchor='center',
            lineWidth=4.0,
            pos = position,
            colorSpace='rgb', lineColor='white', fillColor=None,
            depth=-4.0, interpolate=True, 
            autoDraw = True)
         
        
        # Run 'Begin Routine' code from prac_box_code
        #if no press is made
        right_image = 'stimuli/box_transparent.png'
        left_image = 'stimuli/box_transparent.png'
        
        #left box is magical
        if cue_resp_switch.keys=='1': 
            if(switch_side_loop.thisN < 5):
                left_image = 'stimuli/box_empty_transparent.png'
            else:
                left_image = 'stimuli/box_coin_transparent.png'
        else:
            if(switch_side_loop.thisN < 5):
                right_image = 'stimuli/box_coin_transparent.png'
            else:
                right_image = 'stimuli/box_empty_transparent.png'
        
        # store start times for switch_select
        switch_select.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        switch_select.tStart = globalClock.getTime(format='float')
        switch_select.status = STARTED
        thisExp.addData('switch_select.started', switch_select.tStart)
        switch_select.maxDuration = None
        # keep track of which components have finished
        switch_selectComponents = switch_select.components
        for thisComponent in switch_select.components:
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
        
        # --- Run Routine "switch_select" ---
        # if trial has changed, end Routine now
        if isinstance(switch_side_loop, data.TrialHandler2) and thisSwitch_side_loop.thisN != switch_side_loop.thisTrial.thisN:
            continueRoutine = False
        switch_select.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *left_switch_selec* updates
            
            # if left_switch_selec is starting this frame...
            if left_switch_selec.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_switch_selec.frameNStart = frameN  # exact frame index
                left_switch_selec.tStart = t  # local t and not account for scr refresh
                left_switch_selec.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_switch_selec, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_switch_selec.started')
                # update status
                left_switch_selec.status = STARTED
                left_switch_selec.setAutoDraw(True)
            
            # if left_switch_selec is active this frame...
            if left_switch_selec.status == STARTED:
                # update params
                pass
            
            # if left_switch_selec is stopping this frame...
            if left_switch_selec.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > left_switch_selec.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    left_switch_selec.tStop = t  # not accounting for scr refresh
                    left_switch_selec.tStopRefresh = tThisFlipGlobal  # on global time
                    left_switch_selec.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_switch_selec.stopped')
                    # update status
                    left_switch_selec.status = FINISHED
                    left_switch_selec.setAutoDraw(False)
            
            # *right_switch_selec* updates
            
            # if right_switch_selec is starting this frame...
            if right_switch_selec.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_switch_selec.frameNStart = frameN  # exact frame index
                right_switch_selec.tStart = t  # local t and not account for scr refresh
                right_switch_selec.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_switch_selec, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_switch_selec.started')
                # update status
                right_switch_selec.status = STARTED
                right_switch_selec.setAutoDraw(True)
            
            # if right_switch_selec is active this frame...
            if right_switch_selec.status == STARTED:
                # update params
                pass
            
            # if right_switch_selec is stopping this frame...
            if right_switch_selec.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > right_switch_selec.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    right_switch_selec.tStop = t  # not accounting for scr refresh
                    right_switch_selec.tStopRefresh = tThisFlipGlobal  # on global time
                    right_switch_selec.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_switch_selec.stopped')
                    # update status
                    right_switch_selec.status = FINISHED
                    right_switch_selec.setAutoDraw(False)
            # Run 'Each Frame' code from prac_left_feedback_2
            
                
            
            
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
                switch_select.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in switch_select.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "switch_select" ---
        for thisComponent in switch_select.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for switch_select
        switch_select.tStop = globalClock.getTime(format='float')
        switch_select.tStopRefresh = tThisFlipGlobal
        thisExp.addData('switch_select.stopped', switch_select.tStop)
        # Run 'End Routine' code from prac_left_feedback_2
        #turn off selection indicator
        pr_selection_indicator.setAutoDraw(False)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if switch_select.maxDurationReached:
            routineTimer.addTime(-switch_select.maxDuration)
        elif switch_select.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        
        # --- Prepare to start Routine "switch_out" ---
        # create an object to store info about Routine switch_out
        switch_out = data.Routine(
            name='switch_out',
            components=[left_box_switch_out, right_box_switch_out],
        )
        switch_out.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        left_box_switch_out.setImage(left_image)
        right_box_switch_out.setImage(right_image)
        # store start times for switch_out
        switch_out.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        switch_out.tStart = globalClock.getTime(format='float')
        switch_out.status = STARTED
        thisExp.addData('switch_out.started', switch_out.tStart)
        switch_out.maxDuration = None
        # keep track of which components have finished
        switch_outComponents = switch_out.components
        for thisComponent in switch_out.components:
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
        
        # --- Run Routine "switch_out" ---
        # if trial has changed, end Routine now
        if isinstance(switch_side_loop, data.TrialHandler2) and thisSwitch_side_loop.thisN != switch_side_loop.thisTrial.thisN:
            continueRoutine = False
        switch_out.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *left_box_switch_out* updates
            
            # if left_box_switch_out is starting this frame...
            if left_box_switch_out.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                left_box_switch_out.frameNStart = frameN  # exact frame index
                left_box_switch_out.tStart = t  # local t and not account for scr refresh
                left_box_switch_out.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(left_box_switch_out, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'left_box_switch_out.started')
                # update status
                left_box_switch_out.status = STARTED
                left_box_switch_out.setAutoDraw(True)
            
            # if left_box_switch_out is active this frame...
            if left_box_switch_out.status == STARTED:
                # update params
                pass
            
            # if left_box_switch_out is stopping this frame...
            if left_box_switch_out.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > left_box_switch_out.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    left_box_switch_out.tStop = t  # not accounting for scr refresh
                    left_box_switch_out.tStopRefresh = tThisFlipGlobal  # on global time
                    left_box_switch_out.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'left_box_switch_out.stopped')
                    # update status
                    left_box_switch_out.status = FINISHED
                    left_box_switch_out.setAutoDraw(False)
            
            # *right_box_switch_out* updates
            
            # if right_box_switch_out is starting this frame...
            if right_box_switch_out.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                right_box_switch_out.frameNStart = frameN  # exact frame index
                right_box_switch_out.tStart = t  # local t and not account for scr refresh
                right_box_switch_out.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(right_box_switch_out, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'right_box_switch_out.started')
                # update status
                right_box_switch_out.status = STARTED
                right_box_switch_out.setAutoDraw(True)
            
            # if right_box_switch_out is active this frame...
            if right_box_switch_out.status == STARTED:
                # update params
                pass
            
            # if right_box_switch_out is stopping this frame...
            if right_box_switch_out.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > right_box_switch_out.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    right_box_switch_out.tStop = t  # not accounting for scr refresh
                    right_box_switch_out.tStopRefresh = tThisFlipGlobal  # on global time
                    right_box_switch_out.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'right_box_switch_out.stopped')
                    # update status
                    right_box_switch_out.status = FINISHED
                    right_box_switch_out.setAutoDraw(False)
            
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
                switch_out.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in switch_out.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "switch_out" ---
        for thisComponent in switch_out.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for switch_out
        switch_out.tStop = globalClock.getTime(format='float')
        switch_out.tStopRefresh = tThisFlipGlobal
        thisExp.addData('switch_out.stopped', switch_out.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if switch_out.maxDurationReached:
            routineTimer.addTime(-switch_out.maxDuration)
        elif switch_out.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
        thisExp.nextEntry()
        
    # completed 10.0 repeats of 'switch_side_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "no_coin_instruc" ---
    # create an object to store info about Routine no_coin_instruc
    no_coin_instruc = data.Routine(
        name='no_coin_instruc',
        components=[summary_text],
    )
    no_coin_instruc.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from probability_sequence_code
    num_gold_coins = 0
    # store start times for no_coin_instruc
    no_coin_instruc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    no_coin_instruc.tStart = globalClock.getTime(format='float')
    no_coin_instruc.status = STARTED
    thisExp.addData('no_coin_instruc.started', no_coin_instruc.tStart)
    no_coin_instruc.maxDuration = None
    # keep track of which components have finished
    no_coin_instrucComponents = no_coin_instruc.components
    for thisComponent in no_coin_instruc.components:
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
    
    # --- Run Routine "no_coin_instruc" ---
    no_coin_instruc.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *summary_text* updates
        
        # if summary_text is starting this frame...
        if summary_text.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
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
        
        # if summary_text is stopping this frame...
        if summary_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > summary_text.tStartRefresh + 2.5-frameTolerance:
                # keep track of stop time/frame for later
                summary_text.tStop = t  # not accounting for scr refresh
                summary_text.tStopRefresh = tThisFlipGlobal  # on global time
                summary_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'summary_text.stopped')
                # update status
                summary_text.status = FINISHED
                summary_text.setAutoDraw(False)
        
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
            no_coin_instruc.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in no_coin_instruc.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "no_coin_instruc" ---
    for thisComponent in no_coin_instruc.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for no_coin_instruc
    no_coin_instruc.tStop = globalClock.getTime(format='float')
    no_coin_instruc.tStopRefresh = tThisFlipGlobal
    thisExp.addData('no_coin_instruc.stopped', no_coin_instruc.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if no_coin_instruc.maxDurationReached:
        routineTimer.addTime(-no_coin_instruc.maxDuration)
    elif no_coin_instruc.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.500000)
    thisExp.nextEntry()
    
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
        # Run 'Begin Routine' code from jitters_code
        #Grab right ITI and ISI for this trial num
        ISI = jitters.loc[(jitters['Run'] == 1) & (jitters['Trial'] == trials.thisN + 1)]["ISI"].item() / 1000
        ITI = jitters.loc[(jitters['Run'] == 1) & (jitters['Trial'] == trials.thisN + 1)]["ITI"].item() / 1000
        
        
        
        
        
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
                theseKeys = cue_resp.getKeys(keyList=['1', '2'], ignoreKeys=["escape"], waitRelease=False)
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
        
        if(num_rewarded == 0 and sequence[exp_correct] == 0 and ((cue_resp.keys == left_button and good_side == "left") or (cue_resp.keys == right_button and good_side == "right"))):
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
        if(cue_resp.keys == left_button and good_side == "left"): 
            if(sequence[exp_correct] == 1):
                num_rewarded = num_rewarded + 1
                num_gold_coins = num_gold_coins + 1
            exp_correct = exp_correct + 1 
            
        
            
        if(cue_resp.keys == right_button and good_side == "right"):
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
            if cue_resp.keys == left_button: 
                position = (-0.3, -0.15)
                selection_indicator.setPos(position)
                selection_indicator.setAutoDraw(True)
            elif cue_resp.keys == right_button:
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
        while continueRoutine:
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
                if tThisFlipGlobal > left_box_response.tStartRefresh + ISI-frameTolerance:
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
                if tThisFlipGlobal > right_box_response.tStartRefresh + ISI-frameTolerance:
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
        # the Routine "cue_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
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
        #if left press
        if cue_resp.keys==left_button: 
            #look back one since exp_correct has been incremented
            if sequence[exp_correct - 1] == 1 and good_side == "left":
                left_image = 'stimuli/box_coin_transparent.png'
            else:
                left_image = 'stimuli/box_empty_transparent.png'
        #if right press
        # load in images of either coin box or empty box conditionally
        elif cue_resp.keys==right_button: #same as above
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
        if cue_resp.keys==left_button:
            thisExp.addData('outcome_image', left_image)
        elif cue_resp.keys==right_button:
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
        while continueRoutine:
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
                if tThisFlipGlobal > fixation_cross.tStartRefresh + ITI-frameTolerance:
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
        # the Routine "fixation" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
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
    thisExp.nextEntry()
    
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
