$(() => {
	$('[classname-caret]').on('click', onClassCaretClick);
	$('[subteam-caret]').on('click', onSubteamCaretClick);
	
	$('#createOrg').on('click', (e) => {
		e.preventDefault();
		var form = new FormSubmission($('#createOrgForm'), '/admin/orgs/create', 'createOrg', {autoRetry: false});
		form.submit((err, response) => {
			if (!response) {
				NotificationCard.error('Error: Check the console');
				console.error(err);
			}
			else if (response.status === 200) {
				// On success, 
				NotificationCard.good(response.message);
				setTimeout(() => {
					location.reload();
				}, 1000);
			}
			else {
				NotificationCard.error(response.message);
			}
		});
	});
});

async function deleteOrg(key: string) {
	const result = await PasswordPrompt.show('*DANGER ZONE!!!* Deleting this org will also delete ALL OF ITS USERS.\nIf you are SURE you want to proceed, type your password.');
	if (result.cancelled === false) {
		$.post('/admin/orgs/delete', {password: result.password, org_key: key})
			.done(result => {
				if (result.status === 200) {
					NotificationCard.good(result.message);
					setTimeout(() => {
						location.reload();
					}, 1000);
				}
				else {
					NotificationCard.error(result.message);
				}
			});
	}
}

// Log in to an org's "scoutradioz_admin" user, similar to "sudo su <user>"
async function loginToOrg(key: string) {
	// Prompt for user's password
	const result = await PasswordPrompt.show(`To log into the org ${key}, enter your password. You must be logged in as a "real" Scoutradioz Admin user, with an actual password, to do this.`);
	if (result.cancelled === false) {
		// Send a POST request with their password & the specified org key
		$.post('/admin/orgs/login-to-org', {password: result.password, org_key: key})
			.done(result => {
				// If the status sent back is 200, that means they've successfully logged in as scoutradioz_admin
				if (result.status === 200) {
					location.href = '/home';
				}
				else {
					NotificationCard.error(result.message);
				}
			});
	}
}

function onClassCaretClick(this: HTMLElement, e: Event) {
	if (!e.target) return;
	var elem = $(this);
	var thisParent = elem.parent();
	var aboveParent = thisParent.prev();
	if (aboveParent.length === 0) return console.log('Top class in the list; not doing anything');
	
	var thisIdxStr = (thisParent.attr('id')?.split('_')[1]);
	if (!thisIdxStr) return console.log('Index not defined');
	var thisIdx = parseInt(thisIdxStr);
	var aboveIdx = thisIdx - 1;
	
	// Get the elements for this row & above row
	var thisLabel = thisParent.find(`input[name=classes_${thisIdx}_label]`);
	var thisKey = thisParent.find(`input[name=classes_${thisIdx}_classkey]`);
	var thisSeniority = thisParent.find(`input[name=classes_${thisIdx}_seniority]`);
	var thisYouth = thisParent.find(`select[name=classes_${thisIdx}_youth]`);
	
	var aboveLabel = aboveParent.find(`input[name=classes_${aboveIdx}_label]`);
	var aboveKey = aboveParent.find(`input[name=classes_${aboveIdx}_classkey]`);
	var aboveSeniority = aboveParent.find(`input[name=classes_${aboveIdx}_seniority]`);
	var aboveYouth = aboveParent.find(`select[name=classes_${aboveIdx}_youth]`);
	
	// Cache the values of one of them
	var thisLabelVal = thisLabel.val() || '';
	var thisKeyVal = thisKey.val() || '';
	var thisSeniorityVal = thisSeniority.val() || '';
	var thisYouthVal = thisYouth.val() || '';
	
	// Now, swap them
	thisLabel.val(aboveLabel.val() || '');
	thisKey.val(aboveKey.val() || '');
	thisSeniority.val(aboveSeniority.val() || '');
	thisYouth.val(aboveYouth.val() || '');
	
	aboveLabel.val(thisLabelVal);
	aboveKey.val(thisKeyVal);
	aboveSeniority.val(thisSeniorityVal);
	aboveYouth.val(thisYouthVal);
}

function onSubteamCaretClick(this: HTMLElement, e: Event) {
	if (!e.target) return;
	var elem = $(this);
	var thisParent = elem.parent();
	var aboveParent = thisParent.prev();
	if (aboveParent.length === 0) return console.log('Top class in the list; not doing anything');
	
	var thisIdxStr = (thisParent.attr('id')?.split('_')[1]);
	if (!thisIdxStr) return console.log('Index not defined');
	var thisIdx = parseInt(thisIdxStr);
	var aboveIdx = thisIdx - 1;
	
	console.log('sdffdsfds')
	
	// Get the elements for this row & above row
	var thisLabel = thisParent.find(`input[name=subteams_${thisIdx}_label]`);
	var thisKey = thisParent.find(`input[name=subteams_${thisIdx}_subteamkey]`);
	var thisPitscout = thisParent.find(`select[name=subteams_${thisIdx}_pitscout]`);
	
	var aboveLabel = aboveParent.find(`input[name=subteams_${aboveIdx}_label]`);
	var aboveKey = aboveParent.find(`input[name=subteams_${aboveIdx}_subteamkey]`);
	var abovePitscout = aboveParent.find(`select[name=subteams_${aboveIdx}_pitscout]`);
	
	// Cache the values of one of them
	var thisLabelVal = thisLabel.val() || '';
	var thisKeyVal = thisKey.val() || '';
	var thisPitscoutVal = thisPitscout.val() || '';
	
	thisLabel.val(aboveLabel.val() || '');
	thisKey.val(aboveKey.val() || '');
	thisPitscout.val(abovePitscout.val() || '');
	
	aboveLabel.val(thisLabelVal);
	aboveKey.val(thisKeyVal);
	abovePitscout.val(thisPitscoutVal);
}

function addClass(orgKey: string) {
	//Find out what to name this index
	for (var i = 0; i < 100; i++) {
		if ($(`#classes_${orgKey} #classname_${i}`).length == 0) {
			break;
		}
	}
	var newIdx = i;
	
	var newClass = $('#classTemplate')
		.children()
		.clone();
	newClass.attr('id', `classname_${newIdx}`)
		.appendTo(`#classes_${orgKey}`)
		.find('[classname-caret]')
		.on('click', onClassCaretClick);
	
	newClass.find('input[name=classes_num_label]').attr('name', `classes_${newIdx}_label`);
	newClass.find('input[name=classes_num_classkey]').attr('name', `classes_${newIdx}_classkey`);
	newClass.find('input[name=classes_num_seniority]').attr('name', `classes_${newIdx}_seniority`);
	newClass.find('select[name=classes_num_youth]').attr('name', `classes_${newIdx}_youth`);
}
function deleteClass(orgKey: string) {
	//find lastmost class
	for (var i = 0; i < 100; i++) {
		if ($(`#classes_${orgKey} #classname_${i}`).length == 0) {
			break;
		}
	}
	var lastIdx = i - 1;
	
	$(`#classes_${orgKey} #classname_${lastIdx}`).remove();
}
function addSubteam(orgKey: string) {
	//Find out what to name this index
	for (var i = 0; i < 100; i++) {
		if ($(`#subteams_${orgKey} #subteam_${i}`).length == 0) {
			break;
		}
	}
	var newIdx = i;
	
	var newClass = $('#subteamTemplate')
		.children()
		.clone();
	newClass.attr('id', `subteam_${newIdx}`)
		.appendTo(`#subteams_${orgKey}`)
		.find('[classname-caret]')
		.on('click', onSubteamCaretClick);
	$('input[name=subteams_num_label]').attr('name', `subteams_${newIdx}_label`);
	$('input[name=subteams_num_subteamkey]').attr('name', `subteams_${newIdx}_subteamkey`);
	$('input[name=subteams_num_pitscout]').attr('name', `subteams_${newIdx}_pitscout`);
}
function deleteSubteam(orgKey: string) {
	//find lastmost subteam
	for (var i = 0; i < 100; i++) {
		if ($(`#subteams_${orgKey} #subteam_${i}`).length == 0) {
			break;
		}
	}
	var lastIdx = i - 1;
	
	$(`#subteams_${orgKey} #subteam_${lastIdx}`).remove();
}