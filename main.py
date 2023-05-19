import torch
import os
from models import StoryEndingGenerator, SentimentAnalysis, to_encode_string, perplexity



os.environ["KMP_DUPLICATE_LIB_OK"] = "True"



# Main
if __name__ == '__main__':
    # dataset = load_dataset('./zgodbice/1.txt')

    # Train the sentiment analysis model
    # train_sentiment_model(None)
    # sentiment_analysis = SentimentAnalysis()
    # sentiment_analysis.read_stories_get_sentiment_json("./fairy_tales")
    
    # basic gpt2
    ending_model = StoryEndingGenerator("./nlp-2-training/test")
    
    # fine-tunned gpt2
    # ending_model = StoryEndingGenerator("./content")
    
    # @param: path to dataset
    # @param: output path of model while training
    # ending_model.train("./fairy_tales", "test", './results/stories_context_new.json')

    context = ["Duke negative, Fawkes positive"]
    story = """They quickly rigged up a rude sled, made of poles, put the head of Toover Hek on it, and drew it off to the knight's castle. There it was exposed, on a sharpened stake of wood, in front of the gate. For a whole week it was the sport of the community, and the lads and maidens danced and sang and all the people rejoiced. After the ogre’s head was taken down, it was set in the ground at the side of a brook, and used for women to stand or kneel on, while washing clothes. In time it was polished as ivory and shone in the sun. As for Heinrich, he hitched up four yoke of oxen, and tying an iron chain around the fir tree trunk, which formed the giant’s club, he dragged it to his barnyard and there had it chopped up. It made a load of firewood which lasted him all winter. Now that the roads were safe for all travelers, Heinrich and Grietje, and the knight, in thankfulness to the Holy Virgin fixed a pretty little shrine to one of the forest trees."""
    prompt = to_encode_string(context, story)
    
    # @param: prompt in form of "context [SEP] story", use to_encode_string
    # @param: skip special tokens when decoding the string
    outputs = ending_model.generate_story(prompt, skip_special=False)
    # print(outputs)

    # evaluation


    # Na tej točki je
    #outputs = {
    #    'prompt': "Duke negative, Fawkes positive [SEP] They quickly rigged up a rude sled, made of poles, put the head of Toover Hek on it, and drew it off to the knight's castle. There it was exposed, on a sharpened stake of wood, in front of the gate. For a whole week it was the sport of the community, and the lads and maidens danced and sang and all the people rejoiced. After the ogre’s head was taken down, it was set in the ground at the side of a brook, and used for women to stand or kneel on, while washing clothes. In time it was polished as ivory and shone in the sun. As for Heinrich, he hitched up four yoke of oxen, and tying an iron chain around the fir tree trunk, which formed the giant’s club, he dragged it to his barnyard and there had it chopped up. It made a load of firewood which lasted him all winter. Now that the roads were safe for all travelers, Heinrich and Grietje, and the knight, in thankfulness to the Holy Virgin fixed a pretty little shrine to one of the forest trees.",
    #    'endings': {
    #        0: " When they reached this place where their king sat with them near St Peter�lea—the church-ground walled by two rows —they discovered themselves astonished because so many beautiful things lay ahead; but when King Froghour saw how closely each stood beside others whom she knew nothing about these days,—he went out into what is now called Córdobaŋna (land beyond sea) without stopping until morning: then Fraus told her everything over again before departing from Castle Burdeau.* But soon after I heard Gerbert talking through some other stories concerning me* who lived here three years ago.—I must tell you something else too--that my father died yesterday last night not long afterwards*. Then we rode along toward Grou [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]  well—A good old man lives only once every year among us! We are never more than twenty miles distant.'\nThe Prince laughed bitterly upon hearing such nonsense during our journey home,' said Gandalf 'but if your Majesty could get away no farther!' And Gimli answered :'You see very nicely indeed Sir Robin Croft looks back",
    #        1: ' But then they heard about something terrible happening between them—the witch who kept their hut empty from morning till night; she went out into country with her husband´¬mother when no other way would be possible because so many men lived near each others\' houses.[Pg [PAD] [PAD] [PAD] [PAD]  little]\n [PAD] [PAD]  and half dwarven-haired man stood beside some stone pillars lying outside two large cairns built by three brothers over whom lay six noblemen[611]. They sat quietly under these rocks until suddenly someone came flying through air\xadly towards wheretheywere sleeping.—The horseman stopped abruptly before its owner: "Where is my friend?" The dwarf replied softly enough—"Heaven forbid you should come." So far away did this stranger get sighted within five miles after entering town only once more,—but what frightful thing could happen now? What dreadful fate awaited me! My brother Hans gave himself life upon seeing nothing but death among us strangers like myself every day without being able any longer save those whose hands might burn if I tried hard not even unto tears anymore than ever yet since we entered our house again,[610.] This sudden attack brought terror wherever anyone walked alone behind walls nor saw anything else besides ourselves standing here looking toward heaven above everywhere except——',
    #        2: ' On this altar they buried two golden tablets (for their good fortune) inscribed with names from various ages: The first is St John; by whom we have heard how old these men are now."\nThe King told them nothing about himself but what happened before those times when kings lived under constant fear lest any thing should happen unto others—but then some other woman said something else—"You shall never again see me like you did," says she.—and so I went home after dinner till noon last night,—then came Fraulein back into her room where Mrs Maudlin lay sitting alone beside us laughing loudly enough not even looking out through curtains who looked over every door-frame full size upon our window windows without seeing anything except dark clouds rushing towards ours! But soon afterwards my wife found herself very uneasy because no longer could anybody look near —she felt rather ashamed than [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] —with each step toward happiness more often brought nearer only sorrowing pain still less satisfactionment among ourselves until finally everyone became merry together once better acquainted themselves With such joyous things indeed That none can ever be happier Than thou art happy!" And thus began Marie Boudicherne de Rouxaua——the great poetess',
    #        3: ' When they returned from their travels across Europe this beautiful thing stood upright above them:—It is still standing here today.—I wonder how many times I have seen such things so magnificent! What do you think my dear Prince must be doing? You will see me with your own eyes when we get back again; but what shall come next remains till now.\'\n\'What should she eat?\' asked Elsa laughing aloud into her ear-hole."She has not eaten any meat since yesterday," replied Agnes smilingly—"and neither did anybody else before last day nor after these two years.\'" Then Gerda went out hastily behind Hansa who came rushing towards home without stopping herself even if Frodo wished only half way through by means other than hunting himself.-And then Anna laughed quietly enough too.: \'But why would Ælfric stay long alone,\' said Joanna,—who seemed like something very strange indeed--\'\'the hobbits are gone soon.\'\' She answered hurriedingly:"Ólafro doesn\'t want us quite yet.""No!" cried Frauilin coldily "but please keep going fast because everything seems impossible anyway. And let our horses ride [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]',
    #        4: ' The king came home with great pompousness from town—he must have felt quite ashamed by this event; but when we met each other again after supper some days ago I think our guests gave us permission not only what would happen now (for they thought me crazy),but also how happy you could be if your wife should perish alone! So Frau Gerhard died without delay.—Albion said: But who will take pity upon my sister? She has lived much better than most men do,—and her father did too——she does very well indeed —even though she is still young—"I know nothing about anything like these things," answered Albrecht Baur."The King told them so many times beforehand whereupon Joanne sent two knights into their country-town asking whether any man might marry another woman whose name seemed strange enough even among those whom no mortal knew yet.[Pg [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] ]"Joann replied immediately : \'Yes,\' says Priscilla^\'* A stranger entered quietly beside Alice laughing bitterly.\' At last Joan asked why such good luck ever took place amongst friends alike ; before long Erasmus began weeping over Marie de la Rochefortis´ daughter being killed near Dagenham Hall. Her husband stood'
    #    }
    #}

    # 1. perplexity
    perp_score = perplexity(ending_model, outputs)
    print(perp_score)
    # A lower perplexity score indicates that the language model is better at generating text that fits the given
    # sentiment label.
    # Perplexity scores of the generated reviews are compared to those of the original reviews to evaluate their quality

    # 2. check if ending sentiment is opposite
    # 3. self supervised?
