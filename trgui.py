from terra_ai.utils import ModelVisualizer
from terra_ai.trlayers import LayersDef

__version__ = 0.002


class GUIvalidator(ModelVisualizer):
    """ Make validation of model plan """

    def __init__(self):
        super(GUIvalidator, self).__init__()
        self.model_plan = LayersDef()
        self.val_dictionary = {}

    def model_validation(self, m_plan, inp_shape):
        """
        create validation list for GUI

        Args:
            inp_shape:                tuple or int, Example:
                                            (100,100,3) or 32
            m_plan (list of tuples):    list of layers data as indexes and dictionaries
                                        Example:
                                            [(1,9,1,0,0,0),
                                            (2,9,2,0,0,0),
                                            (3,1,3,{'kernel_size': 2, 'padding': 'same'},2,0),
                                            (4,1,3,3,3,0),
                                            (5,1,3,1,2,0),
                                            (6,1,3,5,5,0),
                                            (7,4,1,0,4,[6]),
                                            (8,3,2,3,2,0)]

        Returns:
            model_plan (list of tuples): same as input model plan
            val_dictionary (dict):       dictionary where
                                            key - layer index in model plan
                                            value - None if layer is valid or
                                                    dict of layer parameters if layer is non-valid
        """

        self.model_plan.plan = m_plan
        self.model_plan.input_shape = inp_shape

        adv_plan = self._get_model_structure(self.model_plan)

        for i in range(len(adv_plan)):
            if adv_plan[i][7] == 'color="red"':
                self.val_dictionary[i] = adv_plan[i][3]
            else:
                self.val_dictionary[i] = None

        # return adv_plan
        return self.val_dictionary, self.model_plan.plan


if __name__ == '__main__':
    input_shape = (54, 96, 3)
    # пример нерабочего плана
    plan = \
        [(1, 9, 1, 0, 0, 0, 0),
         (2, 3, 4, 2, 2, 1, 0),
         (3, 1, 2, 100, 50, 2, 0),
         (4, 1, 2, 100, {'kernel_size': 2, 'padding': 'same'}, 2, 0),
         (5, 4, 1, 0, 0, 3, [4]),
         (6, 8, 1, 0, 0, 5, 0),
         (7, 1, 1, 3, 0, 6, 0)
         ]

    ep = LayersDef()
    ep.plan = plan
    ep.input_shape = input_shape

    gui = GUIvalidator()
    val, mp = gui.model_validation(plan, input_shape)

    print('\n_____Validation list_____\n', val)
    print('\n_____Return model plan_____\n', mp)
    print()
    print(val[0] is None)

    gui.plot_nnmodel(ep, verbose=2)

    gui.output_shape_calculator()
