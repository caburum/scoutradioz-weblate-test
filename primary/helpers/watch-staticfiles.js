// Nodemon is configured to ignore the public-src directory, and I don't think it can be configured to run an NPM script
//	so instead, just run this script (or npm run watch-staticfiles)

const fs = require('fs');
const path = require('path');
const spawn = require('child_process').spawn;
require('colors');

const pathToPublicSrc = path.join(__dirname, '../public-src');
const pathToLess = path.join(pathToPublicSrc, 'less');
const pathToTs = path.join(pathToPublicSrc, 'ts');
const pathToTsBundled = path.join(pathToPublicSrc, 'ts-bundled');

var child, isWorking, startTime;

function kill() {
	if (child) child.kill();
	child = null;
}

function start(command) {
	isWorking = true;
	startTime = Date.now();
	
	child = spawn('npm', [
		'run',
		command,
	], {shell: true});
	
	child.stdout.on('data', function (data) {
		process.stdout.write(data);
	});
	
	child.stderr.on('data', function (data) {
		process.stdout.write(data);
	});
	
	child.on('exit', function (data) {
		console.log(`Done! [${Date.now() - startTime} ms] (Waiting for files to change...)`.yellow);
		isWorking = false;
	});
}

var timeout;

function init() {
	start('compile-static'); // Start by compiling all static files
	
	const time = 100;
	
	// LESS files
	fs.watch(pathToLess, {recursive: true}, (type, filename) => {
		if (filename.endsWith('.less') && !isWorking) {
			if (timeout) clearTimeout(timeout);
			timeout = setTimeout(() => {
				console.log('A change has been detected. Reloading...'.red + ' [LESS]'.yellow);
				start('compile-less');
			}, time); 
		}
	});
	
	// Individual TS files
	fs.watch(pathToTs, {recursive: true}, (type, filename) => {
		if (filename.endsWith('.ts') && !isWorking) {
			if (timeout) clearTimeout(timeout);
			timeout = setTimeout(() => {
				console.log('A change has been detected. Reloading...'.red + ' [TS]'.yellow);
				start('compile-ts');
			}, time); 
		}
	});
	
	// Bundled TS files
	fs.watch(pathToTsBundled, {recursive: true}, (type, filename) => {
		if (filename.endsWith('.ts') && !isWorking) {
			if (timeout) clearTimeout(timeout);
			timeout = setTimeout(() => {
				console.log('A change has been detected. Reloading...'.red + ' [TS bundled]'.yellow);
				start('compile-ts-bundled');
			}, time); 
		}
	});
	
	// fs.watch(pathToPublicSrc, {recursive: true}, (type, filename) => {
	// 	// Only recompile if LESS or TS files are changed
	// 	if ((filename.endsWith('.ts') || filename.endsWith('.less')) && !isWorking) {
	// 		if (timeout) clearTimeout(timeout);
	// 		console.log(type);
	// 		timeout = setTimeout(() => {
	// 			kill();
	// 			if (filename.endsWith('.ts')) {
	// 				// Only compile TypeScript files
	// 				console.log('A change has been detected. Reloading...'.red + ' [TypeScript]'.yellow);
	// 				start('compile-ts');
	// 			}
	// 			else if (filename.endsWith('.less')) {
	// 				// Only compile LESS files
	// 				console.log('A change has been detected. Reloading...'.red + ' [LESS]'.yellow);
	// 				start('compile-less');
	// 			}
	// 		}, 100);
	// 	}
	// });
}

init();