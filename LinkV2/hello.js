// hello.js
const addon = require('./build/Release/addon');
var hello = function(){
	console.log(addon.hello());	
}
 // 'world'

//hello();

module.exports.hello = hello;