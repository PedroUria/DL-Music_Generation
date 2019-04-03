This subdirectory will contain all the data we will use to train our models. It should be organized by [genre](https://www.google.com/search?q=musical+genres&oq=musical+genres&aqs=chrome.0.0l2j69i60j0l3.1480j0j4&sourceid=chrome&ie=UTF-8), author and style.  

Check section 4.12.2 of the source on the main `README.md` for some potential sources. 

**TODO**: Assign yourselves to at least 2 of the sources mentioned on 4.12.2 and make it known to the group. First make sure the files there can be read following [this example](https://github.com/PedroUria/DL-Music_Generation/blob/master/Data_Preprocessing.ipynb) (i.e, that we get the same encoding format after processing it using `music21`) and then download and tag accordingly. You will see a folder called "classical" for an example of this. Other sources are of course welcome, but we need to tag them too. If you are not sure about the style of some files, just tag them as "unknown". 

For example, [this site](http://piano-midi.de/) contains a lot of MIDI files for classical piano music. Looking at this, maybe organizing by tempo would also be a good idea? Although I think we can get that info using `music21`, so it may not be worth it. TODO: Talk about this. 

We could also do *transposition* to double the size of our datasets, and for other potential additional benefits. Check 4.12.1. I think there is a way to do it in *music21*. TODO: after we get some data, but if it works it that would be awesome. 

Check [this](http://web.mit.edu/music21/doc/usersGuide/usersGuide_08_installingMusicXML.html) for more potential sources in other formats. 


NOTE: 

> How do I go about finding a certain classical work in MIDI format in the Internet?
Some MIDI archives in the Internet contain thousands of classical works. You can find the corresponding links to them on my [Linkpage](http://piano-midi.de/links.htm). **The quality of the pieces, however, may vary considerably**.

Taken from [here](http://piano-midi.de/faq.htm). At least I can say that his work is awesome. 