import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device_type', default="cuda")
parser.add_argument('--dataset',default="wiki")
parser.add_argument('--method',default="UTA")
args = parser.parse_args()

if args.dataset =="wiki":
	starting =["I think science fiction is an amazing genre for anything. Future science, technology, time travel, FTL travel, they're all such interesting concepts.", 
	'Can you imagine the world without internet access? ', 
	'I am just finishing my university course and I will be a qualified pharmacist.  I am so excited about finishing and getting out there into the world of work! Do you work yourself?', 
	'I have homebrewed before. Beer can be fun to craft.', 
	'red hair is rare but looks good sometimes.', 
	"I don't know how to be romantic. I have trouble expressing emotional attraction.", 
	'Hello. I hope you might enjoy or know something about Krav Maga?', 
	'I have been trying for years to become a bank teller.', 
	'I like parachunting or skydiving ', 
	'Hello, do you like true crime books at all?', 
	'Denmark seems like a really cool place to visit', 
	'Hi there, I like to go swimming every Thursday, what do you do with your week', 
	'Hi there, I love chocolate, my favorite brand of chocolate is Hershey coming from my local city of Pennsylvania!', 
	'Divorce laws vary by state and in most countries. Most require a court and a legal process with issues of alimony and child support and visitation. I was divorced in 2001.', 
	"I am sort of bored with life right now.  I can't really think of anything exciting to do.  Do you know what I mean?", 
	'I think Im going to wear my contacts from now on', 
	'I teach in the Kindergarten.', 
	'I would like to try fly fishing ', 
	'I drive a commercial truck that has special equipment on it. ', 
	'I loved My Little Pony toys, made by Hasbro, when I was a child. Did you like them?', 
	"I love skateboarding in my free time. It's a sport where you ride and do tricks on a board", 
	'Burritos are so delicious and full of yummy fillings.', 
	'What do you know about the piano? Great instrument!', 
	'Do you like the malt whisky or grain whisky made in Scotland called Scotch whisky?', 
	'I have a pet snake! Do you like snakes?', 
	'I adore the typically furry, carnivorous mammal, known as cats!', 
	'anything with sauce and cheese on it is a pizza. Anything', 
	'Ive been dating this girl recently, shes really pretty and I like her laugh, But I dont know squat about dating, Have you been on dates? can you give me some advice? ', 
	'Hi there, my favorite color is blue! Do you like one of the primary colors, blue?', 
	"I don't play the piano, but I have always wanted to. It's the best acoustic instrument.", 
	'What is your opinion on hitchhiking? (transportation by asking people for a ride)', 
	"I've always wanted to visit Vermont. Do you know anything about it?", 
	"Do you drink Beer? I'm always the one who ends up buying it. Most of it is brewed with hops.", 
	'Running marathons is one of my favorite exercises.', 
	"I'm a bank teller, what do you do?", 
	'Do you like cheeseburgers? They seem to be as popular now than they ever were.', 
	'I have been engaged in an activity related to creating art for several years without success.', 
	'im such a kid sometimes..`i love the show by trey parker and mattstone, south park.do you like it?', 
	'Do you remember the episode of greys anatomy where part of the hospital blew up?', 
	'A dragon is a legendary created, scaled and fire breathing! I wish I could actually have one', 
	'hello! do you like to go hiking?', 
	"I love going hiking! It's so nice to hike through the countryside and on trails.", 
	'I love the book so much that even as a non christian it makes alot of sense', 
	'I like running , except it can be hard on your knees.', 
	'giant panda is an beautiful animal', 
	"Do you have any pets? It's basically defined as an animal kept primarily for company rather than a working animal.", 
	'The Simpsons have been such a part/refelction of modern American culture. Are you a fan of the show?', 
	'I occassionally volunteer in my spare time by giving blood, improving trout streams, or help at food banks. Do you do any volunteer work?', 
	'I love running marathons, all 26 miles!', 
	"I love to eat sushi but I don't actually know much about it"]
	
elif args.dataset =="reddit":
	starting = ['I keep hearing people repeat that "women don\'t wear X for men, they wear it for other women." True? False? Partly true?',
	"Apparently every guy's dream relaxing evening is a steak and a blowjob, but what's the female version? A massage with chocolate? Personal opinions welcome..", 
	"There is this girl I like, but I can't quite tell if she likes me. How can I tell if she does?", 
	"I'm a guy, As the title says, do they. Are they a problem with hooking up with girls and maybe going even further?", 
	"Just curious. Don't hold back. Is Princess Leia really that common of a fantasy?", 
	"Do men find short hair attractive? I mean **really** short hair... Like, I fohawk it level short... I think it looks cool, but i've gotten some intense judgement from it.", 
	"I've never had anyone to impress [before]. What are your preferences? Suggestions? ", 
	'How much time do you need to handle your natural ambivalence? Thanks in advance. ',
	'The smart money seems to be on short-term deficits reinvested into the economy. Of course, you could offset that by slashing the pentagon / intelligent budget by half. ', 
	'In the beginning was a word, and then more words, and finally, enough words that everyone fit neatly into a category: conservative, liberal, left, right, moderate, progressive.', 
	'peace on Earth and goodwill towards mankind. But since most people are cutting back on Christmas presents in this economy, maybe I could get a START treaty? Thanks!Sincerely,Delias1', 
	'For Christianity I am thinking of using this:"Hey derp, who is your favorite scholar from when Christ was alive?"', 
	'There should now be a sub /r/ to which **High Crimes and other Outrages** is appended.', 
	'What do you think about male sex toys like the fleshlight, and "Fuck Me Silly 3 Mega Masturbator".', 
	'My Girlfriend and I decided to go out on a little stroll in the Shopping plaza near her house when we stumbled on this little gem in Big Lots. ',
	"Does that mean she's empowered or she's a rapist?", 
	'After seeing the post on front page I wondered what /r/feminism thought about this.', 
	'Please, please, please unleash the unholy wrath of the internet on these men and their forum.  Make them run and hide.  But be clever about it.', 
	'Politicians or Police?', 
	'60 U.S. citizens were killed from 2000-2008 at foreign embassies; consulates that were not in war zones around the globe from Tashkent to Syria, Republican outrage? ',
	"Why don't Native Americans revolt and demand their land back? Seriously, why not?", 
	'Interesting reading about the democratic primaries and internal struggle. ', 
	"Apparently Brazeau's problem is he didn't raise enough cash ", 
	'As I understand, Judaism is based on the Old Testament and Christianity follows the New Testament. Can somebody explain the difference / significance of the two parts of the Bible?',
	"It seems that the Hong Kong and US portals of Google aren't working from China at the moment. Anyone know what's going on?", 
	"wire transfer is a nightmare.Any reasonable alternative suggestions welcome!(I'm in China)", 
	"Im moving to Shenyang soon, and i'd like to know if i would be able to take my bow and arrows with me.", 
	'Any particular facial products that have worked for you? Any luck with Chinese traditional medicine?', 
	"He also adamantly stated that he would not vote for Hillary Clinton and he's unsure of if he'd vote for Trump.", 
	'Sitting VPs do get preference over most other candidates, but Hillary is a special case to say the least.',
	"Why don't any of you care about the fact that men nearly always have to pay child support, men nearly always lose custody in a divorce, men nearly always lose half their stuff to women during a divorce.  The fact most of you treat men getting rapped as a joke, and domestic abuse (which is about equal both ways)?", 
	'Am I the only one seeing the anti-trump posts?', 
	'Kudos to Miss Murray', 
	'The legendary researcher Dave McGowan (Programmed to Kill) has just posted his very intelligent exposé and breakdown on the staging of the media-centred amputee victim of the Boston bombing. Very well presented with facts, images and analysis that are definitely worth your time to review and consider.', 
	"Being that I am a young single mom, I've often wondered what guys think about dating them! Stories and advice is appreciated!",
	'In Hebrew the word "God" is "el" and probably means "mighty one", or "strong one". I\'ve often heard people say that anything can be a God. Paul mentioned that of some that "their god is their belly" (Phi 3:19)I\'m curious though what others consider the boundary and definition of the word God. ',
	"I haven't read the book yet, but I did go to school with Greg.  He's pretty solid.  Plus the book is free", 
	"I am very interested in hearing everyone's thoughts. Australia and America have very different views in terms of guns [As some may know, guns were banned in Australia after a mass shooting] what I want to know is, do you guys think (whether you're Australian or not.) that Australians being 'morally outraged' and calling Americans stupid for having guns is culturally insignificant? The reason I ask this is because some people   I have encountered in Australia have merely argued ethics without really taking American politics or cultural relativism into account.I just wanted to know what everyone thought?",
	'There is a new subreddit for discussing liberty oriented candidates of all parties: This is a place for libertarians and Ron Paul supporters to discuss and promote candidates for local, state, and Congressional races.', 
	'Of all things, I always thought that reddit would be banned in China. Well, regardless, I have recently started learning Chinese and I am enjoying it very much, so I came here to ask if anyone knows of any Chinese musicians I should seek to know. I am really interested to know how rock would work itself out in Chinese, being such a tonal language and all that. Any suggestions?', 
	'Whereas masturbation goes for the gold (orgasm), fondling is more innocent and idle. Guys do this all the time and wonder how many girls spend time also fondling themselves...or is fondling the same thing as masturbating for girls?', 
	"I'm writing a paper for a class and I can't find much information on the plan, how well it was carried out, or really any in depth information about this specific plan. Can anyone help me out?", 
	"Both parties snuff out the will of voters on the convention. Trump launches the American Party after the open convention results in a Ryan or Kasich nom. Sanders goes rogue as a Dem-Socialist ticket after he is ahead of HRC but the super delegates deny him. The final insult is they don't give him a speaking slot. The GOPe ticket finds no fertile ground and the emperor hits 270 without batting an eye. ", 
	'I understand how hard it is to resist the obvious jokes. But if you want to be taken seriously at the polls maybe you need to start acting serious about your issue?', 
	'It is a complete conspiracy theory at this point, but do you I k that they are going to spend a disproportionate amount of time on our man Trump?', 
	'If God does grant the odd prayer then we do not have free will.  Yet Christians say we have free will and they also say God provides miracles which imo is interference of free will.  explain.', 
	"With my mom about why Trump is right about everything past and future. Side note: Trump should bring up more often the fact that universal health care wont be necessary because he's going to bring medical bills down substantially", 
	'Follow that nice banner to sign up and register to vote for Trump! I know Maryland and Deleware is a closed primary while Indiana is closed. Get registered and as always MAGA!!',
	'hey men, i have a question for you. taking weight and looks out of the picture (although no, i am not overweight or ugly), would a girl who loves video games and possibly knows more about them than you be a turn off? Also, how about comics and general nerdy things? does that also repel? happy new year to you all!',
	'interesting outfit she is wearing..red and black strange cut of cloth..I think she will shave her head soon..']
else:
	print("wrong dataset")

if args.method =="UTA":
	f = open('./results_'+ args.dataset + '/dial.py|--model|DialoGPT-small|--loss_type|1|--device_type|cuda|--seed|0|--adversarial_step|1.txt', "w")
elif args.method =="UTA-LM":
	f = open('./results_'+ args.dataset + '/dial.py|--model|DialoGPT-small|--loss_type|2|--device_type|cuda|--seed|0|--adversarial_step|1.txt', "w")
	
for i in starting:
	if args.method =="UTA":
		subprocess.call(['python', 'dial.py','--model',"microsoft/DialoGPT-small",'--loss_type','1','--device_type',args.device_type,'--seed','0','--adversarial_step','1','--starting_conv',i],stdout=f)
	elif args.method =="UTA-LM":
		subprocess.call(['python', 'dial.py','--model',"microsoft/DialoGPT-small",'--loss_type','2','--device_type',args.device_type,'--seed','0','--adversarial_step','1','--starting_conv',i],stdout=f)

print("The End")
