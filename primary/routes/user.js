const express = require('express');
const router = express.Router();
const wrap = require('express-async-handler');
const utilities = require('@firstteam102/scoutradioz-utilities');
const bcrypt = require('bcryptjs');
const logger = require('log4js').getLogger('user');
const {matchData: matchDataHelper} = require('@firstteam102/scoutradioz-helpers');

router.all('/*', wrap(async (req, res, next) => {
	//Must remove from logger context to avoid unwanted persistent funcName.
	logger.removeContext('funcName');
	next();
}));

//Redirect to index
router.get('/', wrap(async (req, res) => {
	res.redirect(301, '/');
}));

//no longer used bb
router.get('/selectorg', wrap(async (req, res) =>  {
	
	res.redirect(301, '/');
}));

router.get('/login', wrap(async (req, res) => {
	logger.addContext('funcName', 'login[get]');
	
	logger.debug('ENTER');
	
	//If there is no user logged in, send them to select-org page
	if( !req.user ){
		return res.redirect('/?alert=Please select an organization to sign in to.');
	}
	//If the user logged in is NOT default_user, then send them to index.
	else if( req.user.name != 'default_user' ){
		return res.redirect('/?alert=Please log out before you can sign in to another user.');
	}
	//Otherwise, proceed.
	
	//Get organization that user has picked
	var org_key = req.user.org_key;
	logger.debug(`User's organization: ${org_key}`);
	
	//search for organization in database
	var selectedOrg = await utilities.findOne('orgs', 
		{'org_key': org_key}, {},
		{allowCache: true}
	);
	
	//If organization does not exist, send internal error
	if(!selectedOrg) return res.status(500).send('Invalid organization');
	
	res.render('./user/login', {
		title: `Log In to ${selectedOrg.nickname}`,
		org: selectedOrg,
		redirectURL: req.getFixedRedirectURL()
	});
}));

router.post('/login', wrap(async (req, res) => {
	
	//Redirect to /user/login/select via POST (will preserve request body)
	res.redirect(307, '/user/login/select');
}));

router.post('/login/select', wrap(async (req, res) => {
	logger.addContext('funcName', 'login/select[post]');
	//This URL can only be accessed via a POST method, because it requires an organization's password.
	
	logger.debug('ENTER');
	
	//this can only be accessed if someone has logged in to default_user'
	if( !await req.authenticate( process.env.ACCESS_VIEWER ) ) return null;
	
	//get contents of request and selected organization
	var org_key = req.user.org_key;
	var org_password = req.body.org_password;
	logger.debug(`- ${org_key}`);
	
	
	//Make sure that form is filled
	if(!org_key || !org_password || org_key == '' || org_password == ''){
		return res.redirect('/user/login?alert=Please enter your organization\'s password. To go back, open the navigation on the top left.&rdr=' + req.getFixedRedirectURL());
	}
	
	//If form is filled, then proceed.
	
	//Get org that matches request
	var selectedOrg = await utilities.findOne('orgs', 
		{'org_key': org_key}, {},
		{allowCache: true}
	);
	
	//If organization does not exist, send internal error
	if(!selectedOrg) return res.redirect(500, '/user/selectorg');
	
	var passwordHash = selectedOrg.default_password;
	
	//Compare password to correct hash
	var comparison = await bcrypt.compare( org_password, passwordHash );
	
	//If comparison succeeded, then proceed
	if(comparison == true){
		
		var users = await utilities.find('users', 
			{org_key: org_key, name: {$ne: 'default_user'}}, 
			{sort: {name: 1}},
			{allowCache: true}
		);
				
		res.render('./user/selectuser', {
			title: `Sign In to ${selectedOrg.nickname}`,
			org: selectedOrg,
			users: users,
			org_password: org_password, //Must be passed back to user so they can send org's password back with their request (Avoid dealing with tokens & cookies)
			redirectURL: req.body.redirectURL,
		});
	}
	//If failed, then redirect with alert
	else{
		res.redirect(`/user/login?alert=Incorrect password for organization ${selectedOrg.nickname}&rdr=${req.getFixedRedirectURL()}`);
	}
}));

router.post('/login/withoutpassword', wrap(async (req, res) => {
	logger.addContext('funcName', 'login/withoutpassword[post]');
	
	//This is where /user/login/selectuser sends a request first
	var userID = req.body.user;
	var org_key = req.body.org_key;
	var org_password = req.body.org_password;
	
	logger.debug(`userID=${userID}`);
	
	//If we don't have organization info, redirect user to login
	if(!org_key || !org_password){
		return res.send({
			status: 400,
			redirect_url: '/user/login?alert=Sorry, please re-submit your organization login information.'
		});
	}
	
	//If no user is selected, send an alert message
	if(!userID || userID == ''){
		return res.send({
			status: 400,
			alert: 'Please select a user.'
		});
	}
	
	//Get org that matches request
	var selectedOrg = await utilities.findOne('orgs',
		{'org_key': org_key}, {},
		{allowCache: true}
	);
	
	//If organization does not exist, send internal error
	if(!selectedOrg) return res.redirect(500, '/user/login');
	
	var passwordHash = selectedOrg.default_password;
	
	//Compare password to correct hash
	var comparison = await bcrypt.compare( org_password, passwordHash );
	
	//If password isn't correct for some reason, then cry
	if(!comparison){
		return res.send({
			status: 400,
			redirect_url: '/user/login?alert=Sorry, please re-submit your organization login information.'
		});
	}
	
	//Find user info that matches selected id
	var user = await utilities.findOne('users', {_id: userID});
	
	//if user doesn't exist in database for some reason, then cry
	if(!user){
		return res.send({
			status: 400,
			alert: 'No such user exists'
		});
	}
	
	logger.trace(`user: ${JSON.stringify(user)}`);
	
	//Get role information from database, and compare to access role for a scouter
	var role_key = user.role_key;
	var userRole = await utilities.findOne('roles', 
		{role_key: role_key}, {},
		{allowCache: true}
	);
	
	//If no such role exists, throw an error because there must be one
	if(!userRole) throw new Error(`user.js /login/withoutpassword: No role exists in DB with key ${role_key}`);
	
	//if user's access level is greater than scouter, then a password is required.
	if(userRole.access_level > process.env.ACCESS_SCOUTER){
		
		//if user does not have a password but NEEDS a password, then they will need to create one
		if( user.password == 'default' ){
			res.send({
				status: 200,
				create_password: true
			});
		}
		//if user has a non-default password, then they will need to enter it
		else{
			res.send({
				status: 200,
				password_needed: true
			});
		}
	} 
	else if(userRole.access_level == process.env.ACCESS_SCOUTER){
		
		//First, check if the user has a password that is default
		if( user.password == 'default'){
			
			logger.debug('Logging in scouter');
		
			//If password is default, then we may proceed
			req.logIn(user, function(err){
				
				//If error, then log and return an error
				if(err){ console.error(err); return res.send({status: 500, alert: err}); }
				
				logger.debug('Sending success/password_needed: false');
				logger.info(`${user.name} has logged in`);
				
				var redirectURL;
				//if redirectURL has been passed from another function then send it back
				if (req.body.redirectURL) {
					redirectURL = req.body.redirectURL;
				}
				else {
					redirectURL = '/dashboard';
				}
				
				//now, return succes with redirect to dashboard
				res.send({
					status: 200,
					password_needed: false,
					redirect_url: redirectURL,
				});
			});
		}
		else{
			
			logger.debug('Sending password_needed: true');
			
			//if password is not default, then return with password needed.
			res.send({
				status: 200,
				password_needed: true
			});
		}
	}
	else{
		
		logger.debug('Logging in viewer');
		
		//if access_level < process.env.ACCESS_SCOUTER, then log in user
		req.logIn(user, function(err){
			
			//If error, then log and return an error
			if(err){ console.error(err); return res.send({status: 500, alert: err}); }
			
			logger.info(`${user.name} has logged in`);
			
			//Now, return with redirect_url: '/'
			res.send({
				status: 200,
				password_needed: false,
				redirect_url: '/'
			});
		});
	}
}));

router.post('/login/withpassword', wrap(async (req, res) => {
	logger.addContext('funcName', 'login/withpassword[post]');
	
	var userID = req.body.user;
	var userPassword = req.body.password;
	var org_key = req.body.org_key;
	var org_password = req.body.org_password;
	
	logger.debug(`userID=${userID}`);
	
	//If we don't have organization info, redirect user to login
	if(!org_key || !org_password){
		return res.send({
			status: 400,
			redirect_url: '/user/login?Sorry, please re-submit your organization login information.'
		});
	}
	
	//If no user is selected, send an alert message
	if(!userID || userID == ''){
		return res.send({
			status: 400,
			alert: 'Please select a user.'
		});
	}
	
	//Get org that matches request
	var selectedOrg = await utilities.findOne('orgs', 
		{'org_key': org_key}, {},
		{allowCache: true}
	);
	if(!selectedOrg) return res.redirect(500, '/user/login');
	
	var orgPasswordHash = selectedOrg.default_password;
	
	//Compare password to correct hash
	var orgComparison = await bcrypt.compare( org_password, orgPasswordHash );
	
	//If password isn't correct for some reason, then cry
	if(!orgComparison){
		return res.send({
			status: 400,
			redirect_url: '/user/login?Sorry, please re-submit your organization login information.'
		});
	}
	
	//Find user info that matches selected id
	var user = await utilities.findOne('users', {_id: userID});
	
	//if user doesn't exist in database for some reason, then cry
	if(!user || !user.password){
		return res.send({
			status: 400,
			alert: 'No such user exists'
		});
	}
	
	//Compare passwords
	var userComparison = await bcrypt.compare( userPassword, user.password );
	
	logger.trace(`password comparison:${userComparison}`);
	
	if(userComparison){
		
		logger.debug('Logging in');
		
		//If comparison succeeded, then log in user
		req.logIn(user, async function(err){
			
			//If error, then log and return an error
			if(err){ logger.error(err); return res.send({status: 500, alert: err}); }
			
			var userRole = await utilities.findOne('roles', 
				{role_key: user.role_key},
				{allowCache: true}
			);
			
			var redirectURL;
			
			//Set redirect url depending on user's access level
			if (req.body.redirectURL) redirectURL = req.body.redirectURL;
			else if (userRole.access_level == process.env.ACCESS_GLOBAL_ADMIN) redirectURL = '/admin';
			else if (userRole.access_level == process.env.ACCESS_TEAM_ADMIN) redirectURL = '/manage';
			else if (userRole.access_level == process.env.ACCESS_SCOUTER) redirectURL = '/dashboard';
			else redirectURL = '/home';
			
			logger.info(`${user.name} has logged in with role ${userRole.label} (${userRole.access_level}) and is redirected to ${redirectURL}`);
			
			//send success and redirect
			return res.send({
				status: 200,
				redirect_url: redirectURL
			});
		});
	}
	else{
		
		logger.debug('Login failed');
		
		//If authentication failed, then send alert
		return res.send({
			status: 400,
			alert: 'Incorrect password.'
		});
	}
}));

router.post('/login/createpassword', wrap(async (req, res) =>  {
	logger.addContext('funcName', 'login/createpassword[post]');
	
	var userID = req.body.user;
	var org_key = req.body.org_key;
	var org_password = req.body.org_password;
	var p1 = req.body.newPassword1;
	var p2 = req.body.newPassword2;
	
	logger.info(`Request to create password: ${JSON.stringify(req.body)}`);
	
	//If we don't have organization info, redirect user to login
	if(!org_key || !org_password){
		return res.status.send({
			status: 400,
			redirect_url: '/user/login?Sorry, please re-submit your organization login information.'
		});
	}
	
	//If no user is selected, send an alert message
	if(!userID || userID == ''){
		return res.send({
			status: 400,
			alert: 'Please select a user.'
		});
	}
	
	//Get org that matches request
	var selectedOrg = await utilities.findOne('orgs', 
		{'org_key': org_key}, {},
		{allowCache: true}
	);
	if(!selectedOrg) return res.redirect(500, '/user/login');
	
	var orgPasswordHash = selectedOrg.default_password;
	
	//Compare password to correct hash
	var orgComparison = await bcrypt.compare( org_password, orgPasswordHash );
	
	//If password isn't correct for some reason, then cry
	if(!orgComparison){
		return res.send({
			status: 400,
			redirect_url: '/user/login?Sorry, please re-submit your organization login information.'
		});
	}
	
	//Find user info that matches selected id
	var user = await utilities.findOne('users', {_id: userID});
	
	//if user doesn't exist in database for some reason, then cry
	if(!user){
		return res.send({
			status: 500,
			alert: 'No such user exists'
		});
	}
	
	if(user.password != 'default'){
		return res.send({
			password_needed: true,
			alert: 'Password already exists. Please submit your current password.'
		});
	}
	
	//make sure forms are filled
	if( !p1 || !p2 ){
		return res.send({
			alert: 'Please fill both password forms.'
		});
	}
	if( p1 != p2 ){
		return res.send({
			alert: 'Both new password forms must be equal.'
		});
	}
	
	//Hash new password
	const saltRounds = 10;
	
	var hash = await bcrypt.hash( p1, saltRounds );
	
	var writeResult = await utilities.update('users', {_id: userID}, {$set: {password: hash}});
	
	// logger.debug(`${p1} -> ${hash}`);
	logger.debug('createpassword: ' + JSON.stringify(writeResult, 0, 2));
	
	req.logIn(user, function(err){
		
		if(err) logger.error(err);
		
		res.send({
			redirect_url: '/?alert=Set password successfully.'
		});
	});
}));

/**
 * User page to change your own password.
 * @url /login/changepassword
 * @view /login/changepassword
 *
 */
router.get('/changepassword', wrap(async (req, res) => {
	logger.addContext('funcName', 'changepassword[get]');
	if( !await req.authenticate( process.env.ACCESS_SCOUTER ) ) return;
	
	res.render('./user/changepassword', {
		title: 'Change Password'
	});
}));

//Page to change your own password.
router.post('/changepassword', wrap(async (req, res) => {
	logger.addContext('funcName', 'changepassword[post]');
	if( !await req.authenticate( process.env.ACCESS_SCOUTER ) ) return;
	
	var currentPassword = req.body.currentPassword;
	var p1 = req.body.newPassword1;
	var p2 = req.body.newPassword2;
	
	//make sure forms are filled
	if( !p1 || !p2 ){
		return res.redirect('/user/changepassword?alert=Please enter new password.');
	}
	if( p1 != p2 ){
		return res.redirect('/user/changepassword?alert=Both new password forms must be equal.');
	}
	
	var passComparison;
	
	//if user's password is set to default, then allow them to change their password
	if( req.user.password == 'default'){
		passComparison = true;
	}
	else{
		passComparison = await bcrypt.compare(currentPassword, req.user.password);
	}
	
	if( !passComparison ){
		return res.redirect('/user/changepassword?alert=Current password incorrect.');
	}
	
	//Hash new password
	const saltRounds = 10;
	
	var hash = await bcrypt.hash( p1, saltRounds );
	
	var writeResult = await utilities.update('users', {_id: req.user._id}, {$set: {password: hash}});
	
	logger.debug('changepassword: ' + JSON.stringify(writeResult), true);
	
	res.redirect('/?alert=Changed password successfully.');
}));

//Log out
router.get('/logout', wrap(async (req, res) =>  {
	logger.addContext('funcName', 'logout[get]');
	logger.info('ENTER');
	//Logout works a bit differently now.
	//First destroy session, THEN "log in" to default_user of organization.
	
	if( !req.user ) return res.redirect('/');
	
	var org_key = req.user.org_key;
	
	//destroy session
	req.logout();
	
	//req.session.destroy(async function (err) {
	//	if (err) { return next(err); }
		
	//after current session is destroyed, now re log in to org
	var selectedOrg = await utilities.findOne('orgs', 
		{'org_key': org_key}, {},
		{allowCache: true}
	);
	if(!selectedOrg) return res.redirect(500, '/');
	
	var defaultUser = await utilities.findOne('users', 
		{'org_key': org_key, name: 'default_user'}, {},
		{allowCache: true}
	);
	if(!defaultUser) return res.redirect(500, '/');
	
	
	//Now, log in to defaultUser
	req.logIn(defaultUser, async function(err){
			
		//If error, then log and return an error
		if(err){ console.error(err); return res.send({status: 500, alert: err}); }
		
		//now, once default user is logged in, redirect to index
		res.redirect('/');
	});
}));

//Switch a user's organization
router.get('/switchorg', wrap(async (req, res) => {
	logger.addContext('funcName', 'switchorg[get]');
	
	//This will log the user out of their organization.
	
	//destroy session
	req.logout();
	
	req.session.destroy(async function (err) {
		if (err) return console.log(err);
		
		//clear org_key cookie
		logger.debug('Clearing org_key cookie');
		res.clearCookie('org_key');
		
		//now, redirect to index
		res.redirect('/');
	});
}));

//user preferences
router.get('/preferences', wrap(async (req, res) => {
	logger.addContext('funcName', 'preferences[get]');
	
	//Currently the only user preferneces page we have
	res.redirect('/user/preferences/reportcolumns');
}));

router.get('/preferences/reportcolumns', wrap(async (req, res) =>  {
	logger.addContext('funcName', 'preferences/reportcolumns[get]');
	logger.info('ENTER');
	
	var eventYear = req.event.year;
	var orgKey = req.user.org_key;
	var thisOrg = req.user.org;
	var thisOrgConfig = thisOrg.config;
	var redirectURL = req.getFixedRedirectURL(); //////////////////////////////
	
	// read in the list of form options
	var matchlayout = await utilities.find('layout', 
		{org_key: orgKey, year: eventYear, form_type: 'matchscouting'}, 
		{sort: {'order': 1}},
		{allowCache: true}
	);
	//logger.debug("matchlayout=" + JSON.stringify(matchlayout))
	
	var orgColumnDefaults;
	var orgCols = {};
	//Boolean for the view
	var doesOrgHaveNoDefaults = true;
	
	if (thisOrgConfig.columnDefaults && thisOrgConfig.columnDefaults[''+eventYear]) {
		orgColumnDefaults = thisOrgConfig.columnDefaults[''+eventYear];
		doesOrgHaveNoDefaults = false;
	}
	logger.debug(`orgColumnDefaults=${orgColumnDefaults}`);
	
	if (orgColumnDefaults) {
		var orgColArray = orgColumnDefaults.split(',');
		for (var orgCol of orgColArray) {
			orgCols[orgCol] = orgCol;
		}
	}

	var cookieKey = orgKey + '_' + eventYear + '_cols';
	var savedCols = {};
	var colCookie = req.cookies[cookieKey];

	if (req.cookies[cookieKey]) {
		logger.trace('req.cookies[cookie_key]=' + JSON.stringify(req.cookies[cookieKey]));
	}

	//colCookie = "a,b,ccc,d";
	if (colCookie) {
		var savedColArray = colCookie.split(',');
		for (var savedCol of savedColArray)
			savedCols[savedCol] = savedCol;
	}
	logger.debug('savedCols=' + JSON.stringify(savedCols));

	res.render('./user/preferences/reportcolumns', {
		title: 'Choose Report Columns',
		layout: matchlayout,
		savedCols: savedCols,
		orgCols: orgCols,
		doesOrgHaveNoDefaults: doesOrgHaveNoDefaults,
		matchDataHelper: matchDataHelper,
		redirectURL: redirectURL,
	});
}));

router.post('/preferences/reportcolumns', wrap(async (req, res) => {
	logger.addContext('funcName', 'preferences/reportcolumns[post]');
	logger.info('ENTER');
	
	var eventYear = req.event.year;
	var orgKey = req.user.org_key;
	var cookieKey = orgKey + '_' + eventYear + '_cols';
	
	//2020-04-04 JL: Added redirectURL to take user back to previous page
	var setOrgDefault = false;
	
	logger.trace('req.body=' + JSON.stringify(req.body));
	
	var columnArray = [];
	for (var key in req.body) {
		if (key == 'setOrgDefault') {
			setOrgDefault = true;
		}
		else if (key == 'redirectURL') {
			// 2020-04-04 JL: Added exceptions to redirectURL 
			// 2022-03-09 JL: Removed exceptions to redirectURL to make the behavior more consistent
			//	(currently only home, but made it a regex to make it easier to add more in the future)
			//	/\b(?:home|foo|bar)/;
			// var redirectExceptions = /\b(?:home)/;
			// if (!redirectExceptions.test(req.body.redirectURL)) {
			// 	redirectURL = req.body.redirectURL;
			// }
		}
		else {
			columnArray.push(key);
		}
	}
	
	var columnCookie = columnArray.join(',');
	
	logger.debug('columnCookie=' + columnCookie);
	
	/*
	var first = true;
	var columnCookie = '';
	for (var i in req.body) {
		if (i == 'setOrgDefault')    // see choosecolumns.pug
			setOrgDefault = true;
		else {
			if (first)
				first = false;
			else
				columnCookie += ','; 
			columnCookie += i;
		}
	}
	*/
	

	res.cookie(cookieKey, columnCookie, {maxAge: 30E9});
	
	// setting org defaults? NOTE only for Team Admins and above
	if (setOrgDefault && req.user.role.access_level >= process.env.ACCESS_TEAM_ADMIN) {
		logger.debug('Setting org defaults');
		var thisOrg = await utilities.findOne('orgs', 
			{org_key: orgKey}, {},
			{allowCache: true}
		);
		var thisConfig = thisOrg.config;
		if (!thisConfig) {
			thisConfig = {};
			thisOrg['config'] = thisConfig;
		}
		var theseColDefaults = thisOrg.config.columnDefaults;
		if (!theseColDefaults) {
			theseColDefaults = {};
			thisOrg.config['columnDefaults'] = theseColDefaults;
		}

		// set the defaults for this year
		theseColDefaults[eventYear] = columnCookie;
		
		// update DB
		await utilities.update('orgs', {org_key: orgKey}, {$set: {'config.columnDefaults': theseColDefaults}});
		
	}
	
	var redirectURL = req.getRedirectURL();
	logger.debug(`Redirect: ${redirectURL}`);

	res.redirect(redirectURL + '?alert=Saved column preferences successfully.&type=success&autofade=true');
}));

router.post('/preferences/reportcolumns/clearorgdefaultcols', wrap(async (req, res) => {
	logger.addContext('funcName', 'preferences/reportcolumns/clearorgdefaultcols[post]');
	logger.info('ENTER');
	
	var eventYear = req.event.year;
	var orgKey = req.user.org_key;

	if (req.user.role.access_level >= process.env.ACCESS_TEAM_ADMIN) {
		var thisOrg = await utilities.findOne('orgs', 
			{org_key: orgKey}, {},
			{allowCache: true}
		);
		var thisConfig = thisOrg.config;
		//logger.debug("thisConfig=" + JSON.stringify(thisConfig));
		if (!thisConfig) {
			thisConfig = {};
			thisOrg['config'] = thisConfig;
		}
		var theseColDefaults = thisOrg.config.columnDefaults;
		if (!theseColDefaults) {
			theseColDefaults = {};
			thisOrg.config['columnDefaults'] = theseColDefaults;
		}

		// remove values (if they exist) for the event year
		delete theseColDefaults[eventYear];

		// update DB
		await utilities.update('orgs', {org_key: orgKey}, {$set: {'config.columnDefaults': theseColDefaults}});
	}

	res.redirect('/user/preferences/reportcolumns?alert=Cleared org default columns successfully.&type=success&autofade=true');
}));

module.exports = router;
