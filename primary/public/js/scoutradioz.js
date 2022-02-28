/* eslint-disable no-undef */
/* eslint-disable no-unused-vars */
if(!$){
	console.error('scoutradioz.js error: jQuery not enabled');
}

$(() => {
	return;
	// Test to see if cookies have been blocked. If they have, don't bother showing the cookie message.
	// eslint-disable-next-line no-unreachable
	Cookies.set('testcookie', 'foo');
	var cookiesBlocked = (typeof Cookies.get('testcookie') === 'undefined');
	
	if (Cookies.get('accepted') != 'true' || !cookiesBlocked) {
		var cookiesMessage = $(document.createElement('div'))
			.addClass('w3-card w3-padding-large theme-text-secondary')
			.css({
				position: 'fixed',
				width: '100%',
				left: 0,
				right: 0,
				bottom: 0,
				backgroundColor: 'rgba(255,255,255,0.2)'
			})
			.html('Scoutradioz needs to use a small number of cookiies in order to operate. See <a href="/cookies" class="link">our cookie policy</a> for a full list of cookies that are used.')
			.appendTo(document.body);
		var okBtn = $(document.createElement('button'))
			.addClass('w3-button w3-margin-left')
			.text('OK')
			.css({
				backgroundColor: 'rgba(255,255,255,0.2)'
			})
			.on('click', () => {
				Cookies.set('accepted', 'true', {expires: 1000});
				cookiesMessage.remove();
			})
			.appendTo(cookiesMessage);
	}
});

(() => {
	
	
	const resizeCallbacks = [];
	
	/**
	 * Run a particular piece of code when the window resizes, but after waiting a few ms to reduce processor demand.
	 * @param {function} cb Callback to run on a resize event.
	 */
	window.onResize = function(cb) {
		resizeCallbacks.push(cb);
		cb(); // Run the callback once
	};
	
	let ticking = false;
	$(window).on('resize', () => {
		if (resizeCallbacks.length > 0) {
			if (!ticking) {
				ticking = true;
				setTimeout(() => {
					requestAnimationFrame(() => {
						for (let cb of resizeCallbacks) {
							cb();
						}
					});
				}, 50);
			}
		}
	});
	
	/**
	 * Assert a condition & display an error message to the user.
	 * @param {boolean} condition Condition
	 * @param {string} message Message to display
	 */
	window.assert = function (condition, message) {
		if (!condition) {
			var x = new Error();
			NotificationCard.error(
				`ERROR: ${message}\nPlease report as an issue on our GitHub page.\n\nCall stack: ${x.stack}`, 
				{exitable: true, ttl: 0}
			);
		}
	};
	
	var debugLogger = document.createElement('div');
	$(debugLogger).css({
		'background-color': 'white',
		'color': 'black',
		'z-index': '99',
		'position': 'absolute',
		'top': '0',
		'width': '25%',
		'padding': '8px 16px',
	});
	
	window.debugToHTML = function(message) {
		
		var text;
		
		switch (typeof message) {
			case 'string':
			case 'number':
				text = message;
				break;
			case 'object':
			case 'array':
				text = JSON.stringify(message);
				break;
			default:
				text = message;
		}
		
		//if logger is not already added to document.body, add it now
		if ( !$(debugLogger).parent()[0] ) {
			
			$(document.body).append(debugLogger);
		}
		
		var newTextElem = document.createElement('pre');
		$(newTextElem).text(text);
		
		$(debugLogger).append(newTextElem);
	};
})();

class PasswordPrompt {
	constructor(text) {
		if (typeof text !== 'string') throw new TypeError('PasswordPrompt: text must be string.');
		this.text = text;
		this.animateDuration = 200;
	}
	
	static show(text) {
		var newPrompt = new PasswordPrompt(text);
		var promise = newPrompt.show();
		return promise;
	}
	
	show() {
		var card = $(document.createElement('div'))
			.addClass('password-prompt')
			.css('opacity', 0);
		var content = $(document.createElement('div'))
			.addClass('password-prompt-content w3-mobile w3-card')
			.appendTo(card);
		var text = $(document.createElement('div'))
			.html(this._enrichText())
			.addClass('w3-margin-top')
			.appendTo(content);
		var passwordField = $(document.createElement('input'))
			.attr('type', 'password')
			.addClass('w3-input w3-margin-top')
			.on('keydown', this.onKeyDown.bind(this))
			.appendTo(content);
		var btnParent = $(document.createElement('div'))
			.addClass('w3-right-align')
			.appendTo(content);
		var okBtn = $(document.createElement('button'))
			.addClass('w3-btn theme-submit gear-btn')
			.text('OK')
			.on('click', this.resolve.bind(this))
			.appendTo(btnParent);
		var cancelBtn = $(document.createElement('button'))
			.addClass('w3-btn theme-submit gear-btn')
			.text('Cancel')
			.on('click', this.cancel.bind(this))
			.appendTo(btnParent);
			
		this.passwordField = passwordField;
		this.card = card;
		
		NotificationCard.container().append(card);
		// Borrow the NotificationCard container
		this.darkener = $(document.createElement('div'))
			.addClass('canvas')
			.addClass('theme-darkener')
			.css('opacity', 0)
			.appendTo(NotificationCard.container())
			.on('click', this.cancel.bind(this));
		
		// Fade in
		this.card.animate({opacity: 1}, this.animateDuration);
		this.darkener.animate({opacity: 1}, this.animateDuration);
		
		this.promise = new Promise((resolve, reject) => {
			console.log(this);
			this.resolvePromise = resolve;
		});
		return this.promise;
	}
	
	onKeyDown(e) {
		switch(e.key) {
			case 'Enter': this.resolve(); break;
			case 'Escape': this.cancel(); break;
		}
	}
	
	resolve() {
		this.resolvePromise({cancelled: false, password: this.passwordField.val()});
		this.hide();
	}
	
	cancel() {
		this.resolvePromise({cancelled: true});
		this.hide();
	}
	
	hide() {
		// fade out then remove
		this.darkener.animate({opacity: 0}, this.animateDuration);
		this.card.animate({opacity: 0}, {
			duration: this.animateDuration,
			complete: () => {
				this.card.remove();
				this.darkener.remove();
			}
		});
	}
	
	_enrichText() {
		var text = this.text;
		
		//HTML-encode the text of the notificationcard (for safety)
		var enrichedText = $(document.createElement('span'))
			.text(text);
		
		//Enrich text with predetermined keys
		enrichedText = NotificationCard._enrichWithClosingTags(enrichedText.html(), '*', '<b>', '</b>');
		enrichedText = NotificationCard._enrichWithClosingTags(enrichedText.html(), '_', '<i>', '</i>');
		enrichedText = NotificationCard._enrichWithSelfClosingTags(enrichedText.html(), '\n', '</br>');
		enrichedText = NotificationCard._enrichWithSelfClosingTags(enrichedText.html(), '/n', '</br>');
		
		return enrichedText;
	}
}

function share(orgKey) {
	
	var origin = window.location.origin;
	var pathname = window.location.pathname;
	var search = window.location.search;
	
	//if orgKey is defined, add it to the base of the pathname
	if (orgKey != false) {
		pathname = '/' + orgKey + pathname;
	}
	
	var shareURL = origin + pathname + search;
	
	console.log(shareURL);
	
	// Attempt to use navigator.clipboard.writeText
	if (navigator.clipboard && navigator.clipboard.writeText) {
		
		console.log('Attempting navigator.clipboard.writeText');
		
		navigator.clipboard.writeText(shareURL)
			.then(() => {
				NotificationCard.good('Copied link to clipboard. Share it in an app.');
			})
			.catch(err => {
				//Fallback to DOM copy
				console.log(err);
				copyClipboardDom(shareURL);
			});
	}
	else {
		//Fallback to DOM copy
		console.log('navigator.clipboard.writeText does not exist; falling back to DOM copy');
		copyClipboardDom(shareURL);
	}
}

function copyClipboardDom(text) {
	try {
		
		console.log('Attempting DOM copy');
		
		var shareURLInput = $('#shareURLInput');
		shareURLInput.attr('value', text);
		shareURLInput[0].select();
		shareURLInput[0].setSelectionRange(0, 99999); 
		document.execCommand('copy');
		
		NotificationCard.good('Copied link to clipboard. Share it in an app.');
	}
	catch (err) {
		console.error(err);
		NotificationCard.bad(`Could not copy to clipboard. Error: ${err.message}`);
	}
}