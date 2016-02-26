//index.js

var ipc = require('ipc');
var remote = require('remote');

var closeEl = document.querySelector('.close');
var stitchEl = document.querySelector('.stitchButton');
var image1 = document.querySelector('.image1');
var image2 = document.querySelector('.image2');
var image3 = document.querySelector('.image3');
var result = document.querySelector('.result');

closeEl.addEventListener('click', function(){
	console.log("ipc sending message to close window");
	ipc.send('close-main-window');
});

stitchEl.addEventListener('click', function(){
	console.log("ipc sending message to stitch");
	ipc.send('stitch');
	console.log("image being clicked");
	result.src = remote.getCurrentWindow()._res.resultSrc;
});


window.addEventListener('load', function(){
	console.log("ipc sending message to load image" + remote.getCurrentWindow().test);
	
	//ipc.send('imageLoad', image1);
})

var setup = function(){

	image1.src = remote.getCurrentWindow()._dest.image1Src;
	image2.src = remote.getCurrentWindow()._dest.image2Src;
	image3.src = remote.getCurrentWindow()._dest.image3Src;
}

setup();