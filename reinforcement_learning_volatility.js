/******************** 
 * Rl_Reversal *
 ********************/

import { core, data, sound, util, visual, hardware } from './lib/psychojs-2024.2.4.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'RL_reversal';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'session': '001',
    'restart_from_run': [null, 1, 2, 3],
};

// Start code blocks for 'Before Experiment'
// Run 'Before Experiment' code from probability_sequence_code
import * as random from 'random';
import * as csv from 'csv';

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([-1,-1,-1]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(summary_instructionsRoutineBegin());
flowScheduler.add(summary_instructionsRoutineEachFrame());
flowScheduler.add(summary_instructionsRoutineEnd());
const runsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(runsLoopBegin(runsLoopScheduler));
flowScheduler.add(runsLoopScheduler);
flowScheduler.add(runsLoopEnd);











flowScheduler.add(quitPsychoJS, 'Thank you for your patience.', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, 'Thank you for your patience.', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'stimuli/box_transparent.png', 'path': 'stimuli/box_transparent.png'},
    {'name': 'default.png', 'path': 'https://pavlovia.org/assets/default/default.png'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.INFO);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2024.2.4';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}


var summary_instructionsClock;
var summary_text;
var num_switch_list;
var advance_to_runs;
var get_readyClock;
var get_ready_text;
var advance_press;
var scanner_triggerClock;
var scanner_text;
var key_resp;
var cueClock;
var left_box;
var right_box;
var cue_resp;
var good_side;
var num_rewarded;
var num_switch;
var all_keys;
var cue_responseClock;
var left_box_response;
var right_box_response;
var outcomeClock;
var outcome_left_box;
var outcome_right_box;
var fixationClock;
var fixation_cross;
var run_feedbackClock;
var coins_won_text;
var leftover_time_breakClock;
var fixation_end;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "summary_instructions"
  summary_instructionsClock = new util.Clock();
  summary_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'summary_text',
    text: '1. In the game, there will be two boxes, but only one is magical.\n\n2. Sometimes, the magical box switches sides!\n\n3. Sometimes, even the magical box does not have a coin.\n\n\nTry to collect all the coins!\n',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Run 'Begin Experiment' code from define_switches_code
  num_switch_list = [7, 8, 9, 10, 11, 12, 13, 14, 15];
  Math.random.shuffle(num_switch_list);
  
  // Run 'Begin Experiment' code from probability_sequence_code
  /* Syntax Error: Fix Python code */
  advance_to_runs = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "get_ready"
  get_readyClock = new util.Clock();
  get_ready_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'get_ready_text',
    text: 'Get Ready!',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  advance_press = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "scanner_trigger"
  scanner_triggerClock = new util.Clock();
  scanner_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'scanner_text',
    text: 'waiting for scanner',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "cue"
  cueClock = new util.Clock();
  left_box = new visual.ImageStim({
    win : psychoJS.window,
    name : 'left_box', units : undefined, 
    image : 'stimuli/box_transparent.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 0.3), (- 0.1)], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : 0.0 
  });
  right_box = new visual.ImageStim({
    win : psychoJS.window,
    name : 'right_box', units : undefined, 
    image : 'stimuli/box_transparent.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.3, (- 0.1)], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  cue_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Run 'Begin Experiment' code from good_side_code
  good_side = "right";
  num_rewarded = 0;
  num_switch = 0;
  
  all_keys = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "cue_response"
  cue_responseClock = new util.Clock();
  left_box_response = new visual.ImageStim({
    win : psychoJS.window,
    name : 'left_box_response', units : undefined, 
    image : 'stimuli/box_transparent.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 0.3), (- 0.1)], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  right_box_response = new visual.ImageStim({
    win : psychoJS.window,
    name : 'right_box_response', units : undefined, 
    image : 'stimuli/box_transparent.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.3, (- 0.1)], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "outcome"
  outcomeClock = new util.Clock();
  outcome_left_box = new visual.ImageStim({
    win : psychoJS.window,
    name : 'outcome_left_box', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 0.3), (- 0.1)], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  outcome_right_box = new visual.ImageStim({
    win : psychoJS.window,
    name : 'outcome_right_box', units : undefined, 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.3, (- 0.1)], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  // Initialize components for Routine "fixation"
  fixationClock = new util.Clock();
  fixation_cross = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation_cross',
    text: '+',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "run_feedback"
  run_feedbackClock = new util.Clock();
  coins_won_text = new visual.TextStim({
    win: psychoJS.window,
    name: 'coins_won_text',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Initialize components for Routine "leftover_time_break"
  leftover_time_breakClock = new util.Clock();
  fixation_end = new visual.TextStim({
    win: psychoJS.window,
    name: 'fixation_end',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var summary_instructionsMaxDurationReached;
var num_gold_coins;
var _advance_to_runs_allKeys;
var summary_instructionsMaxDuration;
var summary_instructionsComponents;
function summary_instructionsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'summary_instructions' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    summary_instructionsClock.reset();
    routineTimer.reset();
    summary_instructionsMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from probability_sequence_code
    num_gold_coins = 0;
    
    advance_to_runs.keys = undefined;
    advance_to_runs.rt = undefined;
    _advance_to_runs_allKeys = [];
    psychoJS.experiment.addData('summary_instructions.started', globalClock.getTime());
    summary_instructionsMaxDuration = null
    // keep track of which components have finished
    summary_instructionsComponents = [];
    summary_instructionsComponents.push(summary_text);
    summary_instructionsComponents.push(advance_to_runs);
    
    for (const thisComponent of summary_instructionsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function summary_instructionsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'summary_instructions' ---
    // get current time
    t = summary_instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *summary_text* updates
    if (t >= 0.0 && summary_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      summary_text.tStart = t;  // (not accounting for frame time here)
      summary_text.frameNStart = frameN;  // exact frame index
      
      summary_text.setAutoDraw(true);
    }
    
    
    // *advance_to_runs* updates
    if (t >= 0.0 && advance_to_runs.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      advance_to_runs.tStart = t;  // (not accounting for frame time here)
      advance_to_runs.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { advance_to_runs.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { advance_to_runs.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { advance_to_runs.clearEvents(); });
    }
    
    if (advance_to_runs.status === PsychoJS.Status.STARTED) {
      let theseKeys = advance_to_runs.getKeys({keyList: ['space'], waitRelease: false});
      _advance_to_runs_allKeys = _advance_to_runs_allKeys.concat(theseKeys);
      if (_advance_to_runs_allKeys.length > 0) {
        advance_to_runs.keys = _advance_to_runs_allKeys[_advance_to_runs_allKeys.length - 1].name;  // just the last key pressed
        advance_to_runs.rt = _advance_to_runs_allKeys[_advance_to_runs_allKeys.length - 1].rt;
        advance_to_runs.duration = _advance_to_runs_allKeys[_advance_to_runs_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of summary_instructionsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function summary_instructionsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'summary_instructions' ---
    for (const thisComponent of summary_instructionsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('summary_instructions.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(advance_to_runs.corr, level);
    }
    psychoJS.experiment.addData('advance_to_runs.keys', advance_to_runs.keys);
    if (typeof advance_to_runs.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('advance_to_runs.rt', advance_to_runs.rt);
        psychoJS.experiment.addData('advance_to_runs.duration', advance_to_runs.duration);
        routineTimer.reset();
        }
    
    advance_to_runs.stop();
    // the Routine "summary_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var runs;
function runsLoopBegin(runsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    runs = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 3, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'runs'
    });
    psychoJS.experiment.addLoop(runs); // add the loop to the experiment
    currentLoop = runs;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisRun of runs) {
      snapshot = runs.getSnapshot();
      runsLoopScheduler.add(importConditions(snapshot));
      runsLoopScheduler.add(get_readyRoutineBegin(snapshot));
      runsLoopScheduler.add(get_readyRoutineEachFrame());
      runsLoopScheduler.add(get_readyRoutineEnd(snapshot));
      runsLoopScheduler.add(scanner_triggerRoutineBegin(snapshot));
      runsLoopScheduler.add(scanner_triggerRoutineEachFrame());
      runsLoopScheduler.add(scanner_triggerRoutineEnd(snapshot));
      const trialsLoopScheduler = new Scheduler(psychoJS);
      runsLoopScheduler.add(trialsLoopBegin(trialsLoopScheduler, snapshot));
      runsLoopScheduler.add(trialsLoopScheduler);
      runsLoopScheduler.add(trialsLoopEnd);
      runsLoopScheduler.add(run_feedbackRoutineBegin(snapshot));
      runsLoopScheduler.add(run_feedbackRoutineEachFrame());
      runsLoopScheduler.add(run_feedbackRoutineEnd(snapshot));
      runsLoopScheduler.add(leftover_time_breakRoutineBegin(snapshot));
      runsLoopScheduler.add(leftover_time_breakRoutineEachFrame());
      runsLoopScheduler.add(leftover_time_breakRoutineEnd(snapshot));
      runsLoopScheduler.add(runsLoopEndIteration(runsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


var trials;
function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 10, method: TrialHandler.Method.SEQUENTIAL,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial of trials) {
      snapshot = trials.getSnapshot();
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(cueRoutineBegin(snapshot));
      trialsLoopScheduler.add(cueRoutineEachFrame());
      trialsLoopScheduler.add(cueRoutineEnd(snapshot));
      trialsLoopScheduler.add(cue_responseRoutineBegin(snapshot));
      trialsLoopScheduler.add(cue_responseRoutineEachFrame());
      trialsLoopScheduler.add(cue_responseRoutineEnd(snapshot));
      trialsLoopScheduler.add(outcomeRoutineBegin(snapshot));
      trialsLoopScheduler.add(outcomeRoutineEachFrame());
      trialsLoopScheduler.add(outcomeRoutineEnd(snapshot));
      trialsLoopScheduler.add(fixationRoutineBegin(snapshot));
      trialsLoopScheduler.add(fixationRoutineEachFrame());
      trialsLoopScheduler.add(fixationRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


async function runsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(runs);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function runsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var get_readyMaxDurationReached;
var _advance_press_allKeys;
var leftover_t;
var coin;
var get_readyMaxDuration;
var get_readyComponents;
function get_readyRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'get_ready' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    get_readyClock.reset();
    routineTimer.reset();
    get_readyMaxDurationReached = false;
    // update component parameters for each repeat
    advance_press.keys = undefined;
    advance_press.rt = undefined;
    _advance_press_allKeys = [];
    // Run 'Begin Routine' code from init_runs
    num_gold_coins = 0;
    leftover_t = 0;
    if ((expInfo["restart_from_run"] !== null)) {
        if (((runs.thisN - 1) < expInfo["restart_from_run"])) {
            runs.thisN = expInfo["restart_from_run"];
        }
    }
    coin = Math.random.randint(1, 2);
    if ((coin === 1)) {
        good_side = "left";
    } else {
        good_side = "right";
    }
    
    psychoJS.experiment.addData('get_ready.started', globalClock.getTime());
    get_readyMaxDuration = null
    // keep track of which components have finished
    get_readyComponents = [];
    get_readyComponents.push(get_ready_text);
    get_readyComponents.push(advance_press);
    
    for (const thisComponent of get_readyComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function get_readyRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'get_ready' ---
    // get current time
    t = get_readyClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *get_ready_text* updates
    if (t >= 0.0 && get_ready_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      get_ready_text.tStart = t;  // (not accounting for frame time here)
      get_ready_text.frameNStart = frameN;  // exact frame index
      
      get_ready_text.setAutoDraw(true);
    }
    
    
    // *advance_press* updates
    if (t >= 0.0 && advance_press.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      advance_press.tStart = t;  // (not accounting for frame time here)
      advance_press.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { advance_press.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { advance_press.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { advance_press.clearEvents(); });
    }
    
    if (advance_press.status === PsychoJS.Status.STARTED) {
      let theseKeys = advance_press.getKeys({keyList: ['space'], waitRelease: false});
      _advance_press_allKeys = _advance_press_allKeys.concat(theseKeys);
      if (_advance_press_allKeys.length > 0) {
        advance_press.keys = _advance_press_allKeys[_advance_press_allKeys.length - 1].name;  // just the last key pressed
        advance_press.rt = _advance_press_allKeys[_advance_press_allKeys.length - 1].rt;
        advance_press.duration = _advance_press_allKeys[_advance_press_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of get_readyComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function get_readyRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'get_ready' ---
    for (const thisComponent of get_readyComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('get_ready.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(advance_press.corr, level);
    }
    psychoJS.experiment.addData('advance_press.keys', advance_press.keys);
    if (typeof advance_press.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('advance_press.rt', advance_press.rt);
        psychoJS.experiment.addData('advance_press.duration', advance_press.duration);
        routineTimer.reset();
        }
    
    advance_press.stop();
    // the Routine "get_ready" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var scanner_triggerMaxDurationReached;
var _key_resp_allKeys;
var scanner_triggerMaxDuration;
var scanner_triggerComponents;
function scanner_triggerRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'scanner_trigger' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    scanner_triggerClock.reset();
    routineTimer.reset();
    scanner_triggerMaxDurationReached = false;
    // update component parameters for each repeat
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    psychoJS.experiment.addData('scanner_trigger.started', globalClock.getTime());
    scanner_triggerMaxDuration = null
    // keep track of which components have finished
    scanner_triggerComponents = [];
    scanner_triggerComponents.push(scanner_text);
    scanner_triggerComponents.push(key_resp);
    
    for (const thisComponent of scanner_triggerComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function scanner_triggerRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'scanner_trigger' ---
    // get current time
    t = scanner_triggerClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *scanner_text* updates
    if (t >= 0.0 && scanner_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      scanner_text.tStart = t;  // (not accounting for frame time here)
      scanner_text.frameNStart = frameN;  // exact frame index
      
      scanner_text.setAutoDraw(true);
    }
    
    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }
    
    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['=', 'space'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        key_resp.duration = _key_resp_allKeys[_key_resp_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of scanner_triggerComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function scanner_triggerRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'scanner_trigger' ---
    for (const thisComponent of scanner_triggerComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('scanner_trigger.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        psychoJS.experiment.addData('key_resp.duration', key_resp.duration);
        routineTimer.reset();
        }
    
    key_resp.stop();
    // the Routine "scanner_trigger" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var cueMaxDurationReached;
var _cue_resp_allKeys;
var _all_keys_allKeys;
var cueMaxDuration;
var cueComponents;
function cueRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'cue' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    cueClock.reset(routineTimer.getTime());
    routineTimer.add(1.500000);
    cueMaxDurationReached = false;
    // update component parameters for each repeat
    cue_resp.keys = undefined;
    cue_resp.rt = undefined;
    _cue_resp_allKeys = [];
    // Run 'Begin Routine' code from good_side_code
    if ((num_rewarded === num_switch_list[num_switch])) {
        if ((good_side === "right")) {
            good_side = "left";
        } else {
            if ((good_side === "left")) {
                good_side = "right";
            }
        }
        num_rewarded = 0;
        num_switch = (num_switch + 1);
    }
    
    all_keys.keys = undefined;
    all_keys.rt = undefined;
    _all_keys_allKeys = [];
    psychoJS.experiment.addData('cue.started', globalClock.getTime());
    cueMaxDuration = null
    // keep track of which components have finished
    cueComponents = [];
    cueComponents.push(left_box);
    cueComponents.push(right_box);
    cueComponents.push(cue_resp);
    cueComponents.push(all_keys);
    
    for (const thisComponent of cueComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
var next_1;
function cueRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'cue' ---
    // get current time
    t = cueClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *left_box* updates
    if (t >= 0.0 && left_box.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      left_box.tStart = t;  // (not accounting for frame time here)
      left_box.frameNStart = frameN;  // exact frame index
      
      left_box.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (left_box.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      left_box.setAutoDraw(false);
    }
    
    
    // *right_box* updates
    if (t >= 0.0 && right_box.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      right_box.tStart = t;  // (not accounting for frame time here)
      right_box.frameNStart = frameN;  // exact frame index
      
      right_box.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (right_box.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      right_box.setAutoDraw(false);
    }
    
    
    // *cue_resp* updates
    if (t >= 0.0 && cue_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      cue_resp.tStart = t;  // (not accounting for frame time here)
      cue_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { cue_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { cue_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { cue_resp.clearEvents(); });
    }
    
    frameRemains = 0.0 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (cue_resp.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      cue_resp.status = PsychoJS.Status.FINISHED;
        }
      
    if (cue_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = cue_resp.getKeys({keyList: ['1', '2'], waitRelease: false});
      _cue_resp_allKeys = _cue_resp_allKeys.concat(theseKeys);
      if (_cue_resp_allKeys.length > 0) {
        cue_resp.keys = _cue_resp_allKeys[0].name;  // just the first key pressed
        cue_resp.rt = _cue_resp_allKeys[0].rt;
        cue_resp.duration = _cue_resp_allKeys[0].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // Run 'Each Frame' code from good_side_code
    if ((((num_rewarded === 0) && (sequence[exp_correct] === 0)) && (((cue_resp.keys === "1") && (good_side === "left")) || ((cue_resp.keys === "2") && (good_side === "right"))))) {
        console.log("trying to switch");
        sequence[exp_correct] = 1;
        next_1 = util.index(sequence, 1((trials.thisN + 1)));
        console.log(("next one " + next_1.toString()));
        sequence[next_1] = 0;
    }
    
    
    // *all_keys* updates
    if (t >= 0.0 && all_keys.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      all_keys.tStart = t;  // (not accounting for frame time here)
      all_keys.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { all_keys.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { all_keys.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { all_keys.clearEvents(); });
    }
    
    frameRemains = 0.0 + 1.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (all_keys.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      all_keys.status = PsychoJS.Status.FINISHED;
        }
      
    if (all_keys.status === PsychoJS.Status.STARTED) {
      let theseKeys = all_keys.getKeys({keyList: [], waitRelease: false});
      _all_keys_allKeys = _all_keys_allKeys.concat(theseKeys);
      if (_all_keys_allKeys.length > 0) {
        all_keys.keys = _all_keys_allKeys.map((key) => key.name);  // storing all keys
        all_keys.rt = _all_keys_allKeys.map((key) => key.rt);
        all_keys.duration = _all_keys_allKeys.map((key) => key.duration);
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of cueComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function cueRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'cue' ---
    for (const thisComponent of cueComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('cue.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(cue_resp.corr, level);
    }
    psychoJS.experiment.addData('cue_resp.keys', cue_resp.keys);
    if (typeof cue_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('cue_resp.rt', cue_resp.rt);
        psychoJS.experiment.addData('cue_resp.duration', cue_resp.duration);
        routineTimer.reset();
        }
    
    cue_resp.stop();
    // Run 'End Routine' code from good_side_code
    /* Syntax Error: Fix Python code */
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(all_keys.corr, level);
    }
    psychoJS.experiment.addData('all_keys.keys', all_keys.keys);
    if (typeof all_keys.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('all_keys.rt', all_keys.rt);
        psychoJS.experiment.addData('all_keys.duration', all_keys.duration);
        }
    
    all_keys.stop();
    if (cueMaxDurationReached) {
        cueClock.add(cueMaxDuration);
    } else {
        cueClock.add(1.500000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var cue_responseMaxDurationReached;
var selection_indicator;
var too_slow_text;
var remaining_t;
var position;
var cue_responseMaxDuration;
var cue_responseComponents;
function cue_responseRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'cue_response' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    cue_responseClock.reset(routineTimer.getTime());
    routineTimer.add(1.000000);
    cue_responseMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from cue_resp_feedback_code
    selection_indicator = new visual.Rect({"win": psychoJS.window, "name": "polygon", "width": [0.5, 0.5][0], "height": [0.5, 0.5][1], "ori": 0.0, "draggable": false, "anchor": "center", "lineWidth": 4.0, "colorSpace": "rgb", "lineColor": "white", "fillColor": null, "depth": (- 4.0), "interpolate": true, "autoDraw": false});
    too_slow_text = new visual.TextStim({"win": psychoJS.window, "name": "too_slow_text", "text": "too slow", "font": "Arial", "pos": [0, 0.1], "draggable": false, "height": 0.1, "wrapWidth": null, "ori": 0.0, "color": "white", "colorSpace": "rgb", "opacity": 1.0, "languageStyle": "LTR", "depth": (- 6.0), "autoDraw": false});
    if (cue_resp.rt) {
        remaining_t = (1.5 - cue_resp.rt);
        leftover_t = (leftover_t + remaining_t);
        if ((cue_resp.keys === "1")) {
            position = [(- 0.3), (- 0.2)];
            selection_indicator.setPos(position);
            selection_indicator.setAutoDraw(true);
        } else {
            if ((cue_resp.keys === "2")) {
                position = [0.3, (- 0.2)];
                selection_indicator.setPos(position);
                selection_indicator.setAutoDraw(true);
            }
        }
    } else {
        too_slow_text.setAutoDraw(true);
    }
    
    psychoJS.experiment.addData('cue_response.started', globalClock.getTime());
    cue_responseMaxDuration = null
    // keep track of which components have finished
    cue_responseComponents = [];
    cue_responseComponents.push(left_box_response);
    cue_responseComponents.push(right_box_response);
    
    for (const thisComponent of cue_responseComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function cue_responseRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'cue_response' ---
    // get current time
    t = cue_responseClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *left_box_response* updates
    if (t >= 0.0 && left_box_response.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      left_box_response.tStart = t;  // (not accounting for frame time here)
      left_box_response.frameNStart = frameN;  // exact frame index
      
      left_box_response.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (left_box_response.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      left_box_response.setAutoDraw(false);
    }
    
    
    // *right_box_response* updates
    if (t >= 0.0 && right_box_response.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      right_box_response.tStart = t;  // (not accounting for frame time here)
      right_box_response.frameNStart = frameN;  // exact frame index
      
      right_box_response.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 1 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (right_box_response.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      right_box_response.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of cue_responseComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function cue_responseRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'cue_response' ---
    for (const thisComponent of cue_responseComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('cue_response.stopped', globalClock.getTime());
    // Run 'End Routine' code from cue_resp_feedback_code
    selection_indicator.setAutoDraw(false);
    too_slow_text.setAutoDraw(false);
    
    if (cue_responseMaxDurationReached) {
        cue_responseClock.add(cue_responseMaxDuration);
    } else {
        cue_responseClock.add(1.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var outcomeMaxDurationReached;
var outcomeMaxDuration;
var outcomeComponents;
function outcomeRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'outcome' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    outcomeClock.reset(routineTimer.getTime());
    routineTimer.add(1.000000);
    outcomeMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from box_selection_code
    /* Syntax Error: Fix Python code */
    outcome_left_box.setImage(left_image);
    outcome_right_box.setImage(right_image);
    psychoJS.experiment.addData('outcome.started', globalClock.getTime());
    outcomeMaxDuration = null
    // keep track of which components have finished
    outcomeComponents = [];
    outcomeComponents.push(outcome_left_box);
    outcomeComponents.push(outcome_right_box);
    
    for (const thisComponent of outcomeComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function outcomeRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'outcome' ---
    // get current time
    t = outcomeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *outcome_left_box* updates
    if (t >= 0.0 && outcome_left_box.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      outcome_left_box.tStart = t;  // (not accounting for frame time here)
      outcome_left_box.frameNStart = frameN;  // exact frame index
      
      outcome_left_box.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (outcome_left_box.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      outcome_left_box.setAutoDraw(false);
    }
    
    
    // *outcome_right_box* updates
    if (t >= 0.0 && outcome_right_box.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      outcome_right_box.tStart = t;  // (not accounting for frame time here)
      outcome_right_box.frameNStart = frameN;  // exact frame index
      
      outcome_right_box.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (outcome_right_box.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      outcome_right_box.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of outcomeComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function outcomeRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'outcome' ---
    for (const thisComponent of outcomeComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('outcome.stopped', globalClock.getTime());
    // Run 'End Routine' code from box_selection_code
    if ((cue_resp.keys === "1")) {
        psychoJS.experiment.addData("outcome_image", left_image);
    } else {
        if ((cue_resp.keys === "2")) {
            psychoJS.experiment.addData("outcome_image", right_image);
        } else {
            psychoJS.experiment.addData("outcome_image", "no selection made");
        }
    }
    
    if (outcomeMaxDurationReached) {
        outcomeClock.add(outcomeMaxDuration);
    } else {
        outcomeClock.add(1.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var fixationMaxDurationReached;
var fixationMaxDuration;
var fixationComponents;
function fixationRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'fixation' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    fixationClock.reset(routineTimer.getTime());
    routineTimer.add(2.500000);
    fixationMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('fixation.started', globalClock.getTime());
    fixationMaxDuration = null
    // keep track of which components have finished
    fixationComponents = [];
    fixationComponents.push(fixation_cross);
    
    for (const thisComponent of fixationComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function fixationRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'fixation' ---
    // get current time
    t = fixationClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *fixation_cross* updates
    if (t >= 0.0 && fixation_cross.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fixation_cross.tStart = t;  // (not accounting for frame time here)
      fixation_cross.frameNStart = frameN;  // exact frame index
      
      fixation_cross.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 2.5 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (fixation_cross.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      fixation_cross.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of fixationComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function fixationRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'fixation' ---
    for (const thisComponent of fixationComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('fixation.stopped', globalClock.getTime());
    if (fixationMaxDurationReached) {
        fixationClock.add(fixationMaxDuration);
    } else {
        fixationClock.add(2.500000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var run_feedbackMaxDurationReached;
var run_feedbackMaxDuration;
var run_feedbackComponents;
function run_feedbackRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'run_feedback' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    run_feedbackClock.reset(routineTimer.getTime());
    routineTimer.add(4.000000);
    run_feedbackMaxDurationReached = false;
    // update component parameters for each repeat
    coins_won_text.setText((("Congratulations, you collected " + num_gold_coins.toString()) + " gold coins!!"));
    psychoJS.experiment.addData('run_feedback.started', globalClock.getTime());
    run_feedbackMaxDuration = null
    // keep track of which components have finished
    run_feedbackComponents = [];
    run_feedbackComponents.push(coins_won_text);
    
    for (const thisComponent of run_feedbackComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function run_feedbackRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'run_feedback' ---
    // get current time
    t = run_feedbackClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *coins_won_text* updates
    if (t >= 0.0 && coins_won_text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      coins_won_text.tStart = t;  // (not accounting for frame time here)
      coins_won_text.frameNStart = frameN;  // exact frame index
      
      coins_won_text.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 4.0 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (coins_won_text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      coins_won_text.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of run_feedbackComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function run_feedbackRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'run_feedback' ---
    for (const thisComponent of run_feedbackComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('run_feedback.stopped', globalClock.getTime());
    if (run_feedbackMaxDurationReached) {
        run_feedbackClock.add(run_feedbackMaxDuration);
    } else {
        run_feedbackClock.add(4.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var leftover_time_breakMaxDurationReached;
var leftover_time_breakMaxDuration;
var leftover_time_breakComponents;
function leftover_time_breakRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'leftover_time_break' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    leftover_time_breakClock.reset();
    routineTimer.reset();
    leftover_time_breakMaxDurationReached = false;
    // update component parameters for each repeat
    fixation_end.setText((("Take a short brek for  " + leftover_t.toString()) + " seconds!!"));
    psychoJS.experiment.addData('leftover_time_break.started', globalClock.getTime());
    leftover_time_breakMaxDuration = null
    // keep track of which components have finished
    leftover_time_breakComponents = [];
    leftover_time_breakComponents.push(fixation_end);
    
    for (const thisComponent of leftover_time_breakComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function leftover_time_breakRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'leftover_time_break' ---
    // get current time
    t = leftover_time_breakClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *fixation_end* updates
    if (t >= 0.0 && fixation_end.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      fixation_end.tStart = t;  // (not accounting for frame time here)
      fixation_end.frameNStart = frameN;  // exact frame index
      
      fixation_end.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + leftover_t - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (fixation_end.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      fixation_end.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of leftover_time_breakComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function leftover_time_breakRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'leftover_time_break' ---
    for (const thisComponent of leftover_time_breakComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('leftover_time_break.stopped', globalClock.getTime());
    // the Routine "leftover_time_break" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
