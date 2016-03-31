// hello.js
const addon = require('./build/Release/addon');
var hello = function(image1, image2, image3){
	var imageBuffer = addon.hello(image1, image2, image3);
	console.log("Got Buffer");
	console.log(imageBuffer);
	return imageBuffer;	
}
 // 'world'

//hello();

module.exports.hello = hello;