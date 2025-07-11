from typing import List
import numpy as np
from ase import Atoms


class ModelBaseClass():

    """Implementation of Model Base Class
    model needs to have a descriptor which is called in the FALCON main code as model.descriptor
    e.g. if one used the dscibe library: 
    self.descriptor=
    
    """


    def __init__(self):
        pass



    @property
    @abstractmethod
    def implemented_properties(self) -> list[str]:  # pragma: no cover
        """:obj: `list` of :obj: `str`: Implemented properties.
        Available properties are: 'energy', 'forces', 'uncertainty'

        Must be implemented in child class.
        """
        ...




    @abstractmethod
    def predict_energy(self, atoms: Atoms, **kwargs) -> float:  # pragma: no cover
        """Method for energy prediction.

        Note
        ----------
        Always include **kwargs when implementing this function.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object for which to predict the energy.

        Returns
        ----------
        E : float
            The energy prediction

        Must be implemented in child class.
        """

        # E = my_energy_prediction()

        return E
        



    @abstractmethod
    def train(self, training_data: List[Atoms], **kwargs) -> None:  # pragma: no cover
        """Method for model training.

        Note
        ----------
        Always include **kwargs when implementing this function.
        If your model is not trainable just write a method that does nothing

        Parameters
        ----------
        atoms : :obj: `list` of :obj: `ASE Atoms`
            List of ASE atoms objects to use as training data.
            All atoms must have a calculator with energy and other necessary properties set, such that
            it can be accessed by .get_* methods on the atoms.


        Must be implemented in child class.

        """

        # my_model_training()

        pass



    def predict_forces(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """Method for forces prediction.

        Note
        ----------
        Always include **kwargs when implementing this function.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object for which to predict the forces.

        Returns
        ----------
        F : np.array
            The force prediction with shape (N,3), where N is len(atoms)

        """

        # F = my_forces_prediction()
        
        return F



    def predict_uncertainty(self, atoms: Atoms, **kwargs) -> float:
        """Method for energy uncertainty prediction.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object for which to predict the energy.

        Returns
        ----------
        E_unc : float
            The energy uncertainty prediction

        """

        # E_unc = my_energy_uncertainty_prediction()
        
        return E_unc



    def predict_uncertainty_forces(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """Method for energy uncertainty prediction.

        Parameters
        ----------
        atoms : ASE Atoms object
            The atoms object for which to predict the energy.

        Returns
        ----------
        F_unc : np.array
            The force uncertainty prediction with shape (N,3) with N=len(atoms)

        """

        # F_unc = my_forces_uncertainty_prediction()

        return F_unc



class DescriptorBaseClass():

    """Implementation of Descriptor Base Class"""


    def __init__(self):
        pass

    def create_features(atoms: Atoms, **kwargs) -> np.ndarray:
        """
        Method to implement that does the calculation of the feature vector for a single structure.

        This method should not (generally) deal with being given a list of
        Atoms objects.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        features : Features of shape [n_centers, n_features]
        """

        # features = my_feature_vector_calculation()

        return features


    def get_features(atoms: List[Atoms], **kwargs) -> np.ndarray:
        """
        Compute feature tensor for a list of Atoms objects.

        Parameters
        ----------
        atoms : list of ase.Atoms

        Returns
        -------
        features : np.ndarray
            Array of shape [n_structures, n_centers, n_features]
        """

        features = np.array([create_features(a, **kwargs) for a in atoms])

        return features

