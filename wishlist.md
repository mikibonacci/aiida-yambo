# wishlist and improvements for the aiida-yambo package.

The ideal evolution of the package would be to clearly integrate and use the yambopy package as main engine, in the sense of input creation, output parsing, data analysis. This means that in principle the rest of the package is just the conversion from/to AiiDA and yambopy data formats, creation of the AiiDA processes/data and subsmission (the usual AiiDA tasks). The idea would be that it will be possible to initialise the calcjobs and workchains of aiida-yambo from yambopy objects, and viceversa. this will enable to onboard quickly yambopy users on using AiiDA and aiida-yambo, and also will make the aiida-yambo code much lighter and easier (much of the stuff will be in yambopy). 
A first step would be to use as much as yambopy functions and classes as possible. Even creating new AiiDA data structures to store the yambopy data in order to easily convert back into the yambopy corresponding class (example LatticeDB or similar). The AiiDA custom data could be inspired from aiida-vibroscopy. We can even use the yambopy features to generate the list of convergence data and so on... 
Is the high-level-API in aiida-yambo obsolete? i.e., do we have something similar in yambopy or can we in some way improve it?
The idea would be that we can seamlessly go from yambopy data/classes to aiida-yambo and viceversa. At least, as much as possible. 
The idea is to use AiiDA as little as possible, let's say... to run stuff, but then be easily able to open a python shell and from the aiida node obtain yambopy stuff and continue our analysis... 

At this point probably it is worth to notice that some functions, in particular for parsing, are written in aiida-yambo and not in yambopy. the ideal thing would be to add them into yambopy, but for now I would put them in a given module in aiida-yambo, or flag them, and then in the future put them in yambopy, once we are sure they work as expected and are useful also in yambopy. 

once this is in place, I would move to what the workflows does:
- improve the error handling adding more cases and better restarts
- in convergence, possibility to run multidimensional > 3 parameters interdependent convergence, the possibility to choose a given parameter value sets (if not already there, i.e. to run the workflow on a given set of values, e.g. if we just need to do several calculations, not a convergence), and also custom fitting functions.


In yambopy there are many things, for many databases and so on. Maybe some useful database can be retrieved from the aiida calculation as temporary retrieve list, so we can then use yambopy to parse data and store results as AiiDA standard data or new yambopy-like AiiDA data structure. This needs to be understood. 

An additional things we should work on is the ypp stuff. For now there is not much, and also there is no particular post processing workflow implemented. We should consider to do it, with also protocols and get_builder_from_protocols... 