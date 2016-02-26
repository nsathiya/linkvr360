'use strict';

const electron = require('electron');
const app = electron.app;  // Module to control application life.
const BrowserWindow = electron.BrowserWindow;  // Module to create native browser window.
const ipc = require('ipc');
const stitcher = require('./app/proxies/stitcherPX');
//const addon = require('addon');

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
var mainWindow = null;

// Quit when all windows are closed.
app.on('window-all-closed', function() {
  // On OS X it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform != 'darwin') {
    app.quit();
  }
});

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.on('ready', function() {
  // Create the browser window.
  mainWindow = new BrowserWindow({width: 800, height: 800, frame:false, resizeable: false});


  // and load the index.html of the app.
  mainWindow.loadURL('file://' + __dirname + '/app/index.html');
  var _dest = {};
  _dest.image1Src = "../samples/preprocess/CamPic_0.png";
  _dest.image2Src = "../samples/preprocess/CamPic_1.png";
  _dest.image3Src = "../samples/preprocess/CamPic_2.png";

  var _res = {};
  _res.resultSrc = "../samples/preprocess/FinalStitchedResult_0.png";
  
  mainWindow._dest = _dest;
  mainWindow._res = _res;
  // Open the DevTools.
  mainWindow.webContents.openDevTools();

  // Emitted when the window is closed.
  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null;
  });

});

ipc.on('close-main-window', function(){
  console.log("ipc recieving message to close window")
  app.quit();
});

ipc.on('stitch', function(){
  console.log("ipc recieving message to stitch");
  stitcher.startStitching();
    //displaying images
  //stitcher.displayImages();

});

ipc.on('imageLoad', function(event, image){
  console.log("ipc recieving message to load image");
  image.src = "./CamPic_0.png";
  mainWindow.reload();
});