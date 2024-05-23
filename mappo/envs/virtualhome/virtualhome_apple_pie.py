
def init_apple_pie():
    action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the apple pie",  # 4
            "walk to the microwave",  # 5

            "grab the apple pie",  # 6

            "put the apple pie in the microwave",  # 7

            'open the microwave',  # 8
            'close the microwave',  # 9
        ]
    obs2text = obs2text_apple_pie
    return action_template, obs2text


def obs2text_apple_pie(obs, action_template):
    text = ""

    in_kitchen = obs[0]
    in_bathroom = obs[1]
    in_bedroom = obs[2]
    in_livingroom = obs[3]

    see_apple_pie = obs[4]
    close_to_apple_pie = obs[5]
    hold_apple_pie = obs[6]

    see_microwave = obs[7]
    close_to_microwave = obs[8]
    is_microwave_open = obs[9]

    apple_pie_in_microwave = obs[10]

    assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

    # template for room
    in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
    if in_kitchen:
        text += in_room_teplate.format("kitchen")
    elif in_bathroom:
        text += in_room_teplate.format("bathroom")
    elif in_bedroom:
        text += in_room_teplate.format("bedroom")
    elif in_livingroom:
        text += in_room_teplate.format("living room")

    object_text = ""
    action_list = []

    if in_kitchen:

        if not see_apple_pie:
            object_text += "The apple pie is in the microwave. "
        else:
            object_text += "You notice apple pie and microwave. "

        if hold_apple_pie:
            object_text += "Currently, you have grabbed the apple pie in hand. "
            if close_to_microwave:
                object_text += "The microwave is close to you. "
                action_list = [0, 2, 3, 4, 7, 8, 9]
            else:
                object_text += "The microwave is not close to you. "
                action_list = [0, 2, 3, 4, 5]
        else:
            if close_to_apple_pie and not close_to_microwave:
                object_text += "Currently, you are not grabbing anything in hand. The apple pie is close to you. "
                action_list = [0, 2, 3, 5, 6]
            elif close_to_microwave and not close_to_apple_pie:
                object_text += "Currently, you are not grabbing anything in hand. The microwave is close to you. "
                action_list = [0, 2, 3, 4, 8, 9]
            elif not close_to_apple_pie and not close_to_microwave:
                object_text += "Currently, you are not grabbing anything in hand. The apple pie and the microwave are not close to you. "
                action_list = [0, 2, 3, 4, 5]
            else:
                if is_microwave_open:
                    action_list = [0, 2, 3, 8, 9]
                else:
                    action_list = [0, 2, 3, 9]

        if see_apple_pie and is_microwave_open:
            object_text += "The microwave is opened. "
        elif see_apple_pie and not is_microwave_open:
            object_text += "The microwave is not opend. "
        else:
            object_text += "The microwave is closed. "
            action_list = [0, 2, 3]

    elif in_bathroom:

        if hold_apple_pie:
            object_text += "and notice nothing useful. Currently, you have grabbed the apple pie in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [0, 1, 3]
    elif in_bedroom:

        if hold_apple_pie:
            object_text += "and notice nothing useful. Currently, you have grabbed the apple pie in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [0, 1, 2]
    elif in_livingroom:

        if hold_apple_pie:
            object_text += "and notice nothing useful. Currently, you have grabbed the apple pie in hand. "
        else:
            object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

        action_list = [1, 2, 3]

    text += object_text

    target_template = "In order to heat up the apple pie in the microwave, "
    text += target_template

    # template for next step
    next_step_text = "your next step is to"
    text += next_step_text

    actions = [action_template[i] for i in action_list]

    return {"prompt": text, "avaliable_action": actions}