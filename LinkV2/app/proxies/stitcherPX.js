//stitcherProxy.js
//const addon = require('./../../LinkV2/build/Release/addon');

const addon = require('./../../hello');
//var mainDoc = require('../js/index');

module.exports.startStitching = function (){
	console.log("Started Stitching");
	console.log(addon.hello());
	return 0;
}
