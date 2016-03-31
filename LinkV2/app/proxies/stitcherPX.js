//stitcherProxy.js
//const addon = require('./../../LinkV2/build/Release/addon');

const addon = require('./../../hello');
//var mainDoc = require('../js/index');

module.exports.startStitching = function (image1, image2, image3){
	console.log("Started Stitching");
	var _buffImg = addon.hello(image1, image2, image3);
	return _buffImg;
}
