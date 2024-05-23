from mappo.envs.virtualhome.virtualhome_cheese import init_cheese
from mappo.envs.virtualhome.virtualhome_hamburger import init_hamburger
from mappo.envs.virtualhome.virtualhome_apple_pie import init_apple_pie
from mappo.envs.virtualhome.virtualhome_pizza import init_pizza
from mappo.envs.virtualhome.virtualhome_washing_plate import init_washing_plate
from mappo.envs.virtualhome.virtualhome_laundry import init_laundry

def init_variant_env(env_id, variant):
    assert env_id == "VirtualHome-v1", "Only VirtualHome-v1 is supported"
    assert variant in ["Cheese", "Hamburger", "Apple Pie", "Pizza", "Washing Plate", "Laundry"], "Only Cheese, Hamburger, Apple Pie, Pizza, Washing Plate, Laundry are supported"
    
    print("variant: ", variant)
    
    if variant == "Cheese":
        action_template, obs2text = init_cheese()
    elif variant == "Hamburger":
        action_template, obs2text = init_hamburger()
    elif variant == "Apple Pie":
        action_template, obs2text = init_apple_pie()
    elif variant == "Pizza":
        action_template, obs2text = init_pizza()
    elif variant == "Washing Plate":
        action_template, obs2text = init_washing_plate()
    elif variant == "Laundry":
        action_template, obs2text = init_laundry()
    else:
        raise NotImplementedError("Not implemented variant")
    return action_template, obs2text




