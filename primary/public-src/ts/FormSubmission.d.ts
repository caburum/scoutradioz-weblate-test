
declare class FormSubmission{
	
	data: Dictionary<any>;
	url: string;
	key: string;
	/**
	 * Generic form submission that uses NotificationCards.
	 * @param {HTMLFormElement|JQuery} form Form to submit
	 * @param {String} url POST URL to submit to
	 * @param {String} key Name of form (any)
	 * @param {object} [options]
	 * @param {boolean} [options.autoRetry=true] Whether to auto retry.
	 */
	constructor(form: HTMLFormElement|JQuery, url: string, key: string, options?: {autoRetry: boolean});
	
	/**
	 * Submit the formsubmission.
	 * @param {ObjectCallback} cb Callback function. (err, message)
	 */
	submit(cb: ObjectCallback): void;
	
	_getFromLocalStorage(): string;
	
	_addToLocalStorage(): void;
	
	_getFormData(form: HTMLFormElement|JQuery): Dictionary<String>;
}

interface ObjectCallback {
	(error: JQueryXHR|Error|string|null, response?: SRResponse): void;
}

declare class SRResponse {
	status: number;
	message: string;
}

interface Dictionary<T> {
    [Key: string]: T;
}