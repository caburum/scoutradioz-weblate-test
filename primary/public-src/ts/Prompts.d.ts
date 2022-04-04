
declare class PasswordPrompt {
	
	text: string;
	animateDuration: number;
	card: JQuery;
	darkener: JQuery;
	passwordField: JQuery;
	promise: Promise<{cancelled: boolean}> | undefined;
	resolvePromise: Function | undefined;
	
	static show(text: string): Promise<PasswordResponse>;
	
	show(): void;
	
	onKeyDown(e: KeyboardEvent): void;
	
	resolve(): void;
	
	cancel(): void;
	
	hide(): void;
	
	_enrichText(): void;
}


declare class Confirm {
	
	text: string;
	yesText: string;
	noText: string;
	animateDuration: number;
	yesTimeout: number; // Time, in ms, before allowing the user to click "yes"
	card: JQuery;
	darkener: JQuery;
	promise: Promise<{cancelled: boolean}> | undefined;
	resolvePromise: Function | undefined;
	
	static show(text: string): Promise<PromptResponse>;
	
	show(): void;
	onKeyDown(e: KeyboardEvent): void;
	resolve(): void;
	cancel(): void;
	hide(): void;
	_enrichText(): void;
}

declare class ConfirmOptions {
	yesText?: string;
	noText?: string;
	yesTimeout?: number;
}

declare class PromptResponse {
	cancelled: boolean;
}

declare class PasswordResponse extends PromptResponse {
	password: string;
}