"""Data nodes for the Yambo databases which yambopy reads: the lattice (`ns.db1`) and the excitons (`ndb.BS_diago_Q*`).

They keep in the AiiDA repository the arrays yambopy needs, not the netCDF files, and give back the yambopy objects
(`to_yambopy`), so that the analysis (exciton composition, response function...) can be done on the nodes:

    exciton = calculation.outputs.exciton_db.Q1
    yambopy_exciton = exciton.to_yambopy()   # the lattice is found in the outputs of the same calculation
"""
import numpy as np

from aiida.orm import ArrayData

LATTICE_ARRAYS = ('lat', 'alat', 'sym_car', 'iku_kpoints', 'car_atomic_positions', 'atomic_numbers')
LATTICE_INTEGERS = ('time_rev', 'spinor_components', 'nelectrons', 'mag_syms')


def _yambopy_version():
    try:
        from importlib.metadata import version
        return version('yambopy')
    except Exception:  # pylint: disable=broad-except
        return None


class YamboLatticeData(ArrayData):
    """The lattice of a Yambo calculation: the content of `ns.db1` which `YamboLatticeDB` (yambopy) uses.

    Only the k-points of the irreducible Brillouin zone are stored: `to_yambopy` expands them as yambopy does when it
    reads `ns.db1`, so that the indices of the excitonic transitions refer to the same k-points.
    """

    @classmethod
    def from_yambopy(cls, lattice):
        """Create the node from a `YamboLatticeDB`, expanded or not."""
        node = cls()
        # `expand_kpoints` keeps the irreducible k-points here and replaces `iku_kpoints` with the full grid
        iku_kpoints = lattice.iku_kpoints if getattr(lattice, 'ibz_kpoints', None) is None else lattice.ibz_kpoints
        arrays = {name: getattr(lattice, name) for name in LATTICE_ARRAYS}
        arrays['iku_kpoints'] = iku_kpoints
        for name, value in arrays.items():
            node.set_array(name, np.array(value))
        for name in LATTICE_INTEGERS:
            node.base.attributes.set(name, int(getattr(lattice, name)))
        node.base.attributes.set('yambopy_version', _yambopy_version())
        return node

    @classmethod
    def from_netcdf(cls, filename):
        """Create the node from a `ns.db1` file."""
        from yambopy.dbs.latticedb import YamboLatticeDB

        return cls.from_yambopy(YamboLatticeDB.from_db_file(filename=str(filename), Expand=False))

    def to_yambopy(self, expand=True, atol=1e-6):
        """The `YamboLatticeDB` of yambopy. With `expand` the k-points are expanded to the full zone, as `from_db_file` does."""
        from yambopy.dbs.latticedb import YamboLatticeDB

        lattice = YamboLatticeDB(
            **{name: self.get_array(name) for name in LATTICE_ARRAYS},
            **{name: self.base.attributes.get(name) for name in LATTICE_INTEGERS},
        )
        if expand:
            lattice.expand_kpoints(atol=atol, verbose=0)
        return lattice


class YamboExcitonData(ArrayData):
    """The excitons of a BSE calculation at one Q point: the content of `ndb.BS_diago_Q*` which yambopy uses.

    Always stored: the eigenvalues and the left and right residuals. The transitions table (`table`) is there only if
    yambo wrote it (it needs `WRbsWF`), and the eigenvectors only if they were requested (setting
    `STORE_EXCITON_EIGENVECTORS`): they are stored in single precision (complex64) and are the largest part.
    """

    @classmethod
    def from_yambopy(cls, exciton, store_eigenvectors=False, neigs=None):
        """Create the node from a `YamboExcitonDB`.

        :param store_eigenvectors: also store the eigenvectors, if the `YamboExcitonDB` has them.
        :param neigs: store only the eigenvectors of the first `neigs` excitons.
        """
        node = cls()
        node.set_array('eigenvalues', np.asarray(exciton.eigenvalues))
        node.set_array('l_residual', np.asarray(exciton.l_residual))
        node.set_array('r_residual', np.asarray(exciton.r_residual))
        if exciton.table is not None:
            node.set_array('table', np.asarray(exciton.table).astype(int))
        if exciton.car_qpoint is not None:
            node.set_array('car_qpoint', np.asarray(exciton.car_qpoint))
        has_eigenvectors = bool(store_eigenvectors) and exciton.eigenvectors is not None
        if has_eigenvectors:
            node.set_array('eigenvectors', np.asarray(exciton.eigenvectors)[:neigs].astype(np.complex64))

        attributes = {
            'Qpt': str(exciton.Qpt),
            'spin_pol': exciton.spin_pol,
            'Lkind': exciton.Lkind,
            'q_cutoff': None if exciton.q_cutoff is None else float(exciton.q_cutoff),
            'has_table': exciton.table is not None,
            'has_eigenvectors': has_eigenvectors,
            'number_of_excitons': len(exciton.eigenvalues),
            'yambopy_version': _yambopy_version(),
        }
        for name, value in attributes.items():
            node.base.attributes.set(name, value)
        return node

    def get_lattice(self):
        """The `YamboLatticeData` of the calculation which created this node (its `lattice` output)."""
        try:
            return self.creator.outputs.lattice
        except Exception as error:  # pylint: disable=broad-except
            raise ValueError('the lattice is not in the outputs of the calculation which created this node: '
                             'pass it to `to_yambopy`') from error

    def to_yambopy(self, lattice=None, expand=True):
        """The `YamboExcitonDB` of yambopy.

        :param lattice: a `YamboLatticeData` (or a yambopy `YamboLatticeDB`); by default the `lattice` output of the
            calculation which created this node.
        :param expand: expand the k-points of the lattice to the full zone (as yambopy does when reading `ns.db1`).
        """
        from yambopy.dbs.excitondb import YamboExcitonDB

        if lattice is None:
            lattice = self.get_lattice()
        if isinstance(lattice, YamboLatticeData):
            lattice = lattice.to_yambopy(expand=expand)

        arrays = set(self.get_arraynames())
        return YamboExcitonDB(
            lattice,
            self.base.attributes.get('Qpt'),
            self.get_array('eigenvalues'),
            self.get_array('l_residual'),
            self.get_array('r_residual'),
            spin_pol=self.base.attributes.get('spin_pol'),
            car_qpoint=self.get_array('car_qpoint') if 'car_qpoint' in arrays else None,
            q_cutoff=self.base.attributes.get('q_cutoff'),
            Lkind=self.base.attributes.get('Lkind'),
            table=self.get_array('table') if 'table' in arrays else None,
            eigenvectors=self.get_array('eigenvectors') if 'eigenvectors' in arrays else None,
        )
