import unittest
import numpy
import pyscf
from pyscf import lib
from pyscf import neo
from pyscf import gto, dft, grad
from pyscf.data import nist
from pyscf.qmmm.pbc import itrf, mm_mole

def setUpModule():
    global mol, mol_1e6, mol_dft, mol_point, mol_small_radius, box, mm_coords,\
           mm_charges, mm_radii, rcut_ewald, rcut_hcore, mm_mol_pbc, mm_mol_point, mm_mol_small_radius

    box = numpy.eye(3)*12
    mm_coords = [[1,2,-1],[3,4,5]]
    mm_charges = [-0.8,0.8]
    mm_radii = [0.8,1.2]
    rcut_ewald = 8
    rcut_hcore = 6
    mm_mol_pbc = mm_mole.create_mm_mol(mm_coords, box, mm_charges, mm_radii, rcut_ewald=rcut_ewald, rcut_hcore=rcut_hcore)
    mm_mol_point = mm_mole.create_mm_mol(mm_coords, box, mm_charges, rcut_ewald=rcut_ewald, rcut_hcore=rcut_hcore)
    mm_mol_small_radius = mm_mole.create_mm_mol(mm_coords, box, mm_charges, [1e-8]*2, rcut_ewald=rcut_ewald, rcut_hcore=rcut_hcore)
    atom='''
         O       0.0000000000    -0.0000000000     0.1174000000
         H      -0.7570000000    -0.0000000000    -0.4696000000
         H       0.7570000000     0.0000000000    -0.4696000000
         '''
    mol = neo.M(atom=atom, basis='631G', nuc_basis='pb4d',
                quantum_nuc=['H'], mm_mol_pbc=mm_mol_pbc)
    mol_1e6 = neo.M(atom=atom, basis='631G', nuc_basis='1e6',
                    quantum_nuc=['H'], mm_mol_pbc=mm_mol_pbc)
    mol_dft = gto.M(atom=atom, basis='631G')
    mol_point = neo.M(atom=atom, basis='631G', nuc_basis='pb4d',
                      quantum_nuc=['H'], mm_mol_pbc=mm_mol_point)
    mol_small_radius = neo.M(atom=atom, basis='631G', nuc_basis='pb4d',
                             quantum_nuc=['H'], mm_mol_pbc=mm_mol_small_radius)

def tearDownModule():
    global mol, mol_1e6, mol_dft, mol_point, mol_small_radius, box, mm_coords,\
        mm_charges, mm_radii, rcut_ewald, rcut_hcore, mm_mol_pbc, mm_mol_point, mm_mol_small_radius

class KnowValues(unittest.TestCase):
    def test_no_neo(self):
        mf = neo.CDFT(mol_1e6, xc='PBE')
        e_noneo = mf.kernel()
        mass = mol.mass[1] * nist.ATOMIC_MASS / nist.E_MASS
        ke = numpy.einsum('ij,ji->', mol_1e6.components['n1'].intor_symmetric('int1e_kin'), mf.components['n1'].make_rdm1()) / mass
        e_noneo -= 2*ke
        g_noneo = mf.nuc_grad_method()
        g_noneo_qm = g_noneo.kernel()
        g_noneo_mm = g_noneo.grad_mm()

        mf = dft.RKS(mol_dft, xc='PBE')
        mf = itrf.add_mm_charges(mf, mm_coords, box, mm_charges, mm_radii, rcut_ewald, rcut_hcore)
        e_dft = mf.scf()
        g_dft = mf.nuc_grad_method()
        g_dft_qm = g_dft.kernel()
        g_dft_mm = g_dft.grad_mm()
        self.assertAlmostEqual(e_noneo, e_dft, 5)
        numpy.testing.assert_array_almost_equal(g_noneo_qm, g_dft_qm, 6)
        numpy.testing.assert_array_almost_equal(g_noneo_mm, g_dft_mm, 6)

    def test_point_and_small_radius_gaussian(self):
        mf_point = neo.CDFT(mol_point, xc='PBE')
        e_point = mf_point.kernel()
        g_point = mf_point.nuc_grad_method()
        g_point_qm = g_point.kernel()
        g_point_mm = g_point.grad_mm()

        mf_small_radius = neo.CDFT(mol_small_radius, xc='PBE')
        e_small_radius = mf_small_radius.kernel()
        g_small_radius = mf_small_radius.nuc_grad_method()
        g_small_radius_qm = g_small_radius.kernel()
        g_small_radius_mm = g_small_radius.grad_mm()
        self.assertAlmostEqual(e_point, e_small_radius, 8)
        numpy.testing.assert_array_almost_equal(g_point_qm, g_small_radius_qm, 6)
        numpy.testing.assert_array_almost_equal(g_point_mm, g_small_radius_mm, 6)

    def test_finite_difference_gradient(self):
        mf = neo.CDFT(mol, xc='PBE')
        mf.components['e'].grids.atom_grid = (99, 974)
        mf.conv_tol = 1e-12
        mf.kernel()
        g = mf.nuc_grad_method()
        g_qm = g.kernel()
        g_mm = g.grad_mm()

        atom1 = '''
                 O       0.0000000000    -0.0000000000     0.1175000000
                 H      -0.7570000000    -0.0000000000    -0.4696000000
                 H       0.7570000000     0.0000000000    -0.4696000000
                 '''
        mol1 = neo.M(atom=atom1, basis='631G', nuc_basis='pb4d',
                      quantum_nuc=['H'], mm_mol_pbc=mm_mol_pbc)
        mf1 = neo.CDFT(mol1, xc='PBE')
        mf1.components['e'].grids.atom_grid = (99, 974)
        mf1.conv_tol = 1e-12
        e1 = mf1.kernel()

        atom2 = '''
                 O       0.0000000000    -0.0000000000     0.1173000000
                 H      -0.7570000000    -0.0000000000    -0.4696000000
                 H       0.7570000000     0.0000000000    -0.4696000000
                 '''
        mol2 = neo.M(atom=atom2, basis='631G', nuc_basis='pb4d',
                      quantum_nuc=['H'], mm_mol_pbc=mm_mol_pbc)
        mf2 = neo.CDFT(mol2, xc='PBE')
        mf2.components['e'].grids.atom_grid = (99, 974)
        mf2.conv_tol = 1e-12
        e2 = mf2.kernel()

        self.assertAlmostEqual(g_qm[0,2], (e1-e2)/0.0002*lib.param.BOHR, 5)

    def test_ewald_potential(self):
        mf = neo.CDFT(mol, xc='PBE')
        mf.components['e'].grids.atom_grid = (99, 974)
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        quadrupole_n1 = mf.components['n1'].get_qm_quadrupoles(dm['n1'])
        quadrupole_n2 = mf.components['n2'].get_qm_quadrupoles(dm['n2'])
        ewald_pot_qm_cneo = mf.get_qm_ewald_pot(dm=dm)
        mf_dft = dft.RKS(mol_dft, xc='PBE')
        mf_dft = itrf.add_mm_charges(mf_dft, mm_coords, box, mm_charges, mm_radii, rcut_ewald, rcut_hcore)
        def get_qm_ewald_pot(mf, mol, dm, d_quadrupoles=None, qm_ewald_hess=None):
            # hess = d^2 E / dQ_i dQ_j, d^2 E / dQ_i dD_ja, d^2 E / dDia dDjb, d^2 E/ dQ_i dO_jab
            if qm_ewald_hess is None:
                qm_ewald_hess = mf.mm_mol.get_ewald_pot(mol.atom_coords())
                mf.qm_ewald_hess = qm_ewald_hess
            charges = mf.get_qm_charges(dm)
            dips = mf.get_qm_dipoles(dm)
            quads = mf.get_qm_quadrupoles(dm)
            if d_quadrupoles is not None:
                quads += d_quadrupoles
            ewpot0  = lib.einsum('ij,j->i', qm_ewald_hess[0], charges)
            ewpot0 += lib.einsum('ijx,jx->i', qm_ewald_hess[1], dips)
            ewpot0 += lib.einsum('ijxy,jxy->i', qm_ewald_hess[3], quads)
            ewpot1  = lib.einsum('ijx,i->jx', qm_ewald_hess[1], charges)
            ewpot1 += lib.einsum('ijxy,jy->ix', qm_ewald_hess[2], dips)
            ewpot2  = lib.einsum('ijxy,j->ixy', qm_ewald_hess[3], charges)
            return ewpot0, ewpot1, ewpot2
        ewald_pot_qm_dft = get_qm_ewald_pot(mf_dft, mol_dft, dm['e'], d_quadrupoles=quadrupole_n1+quadrupole_n2)
        for ew_pot_cneo, ew_pot_dft in zip(ewald_pot_qm_cneo, ewald_pot_qm_dft):
            numpy.testing.assert_array_almost_equal(ew_pot_cneo, ew_pot_dft, 6)

    def test(self):
        mf = neo.CDFT(mol, xc='PBE')
        mf.components['e'].grids.atom_grid = (99, 974)
        mf.conv_tol = 1e-12
        e = mf.kernel()
        g = mf.nuc_grad_method()
        g_qm = g.kernel()
        g_mm = g.grad_mm()
        self.assertAlmostEqual(e, -76.22383202883007, 8)
        numpy.testing.assert_array_almost_equal(g_qm,\
                numpy.array([[ 0.00406731,  0.02311284, -0.04986   ],
                             [ 0.03681785, -0.00920357,  0.02293419],
                             [-0.04040896, -0.02454246,  0.02187715]]), 6)
        numpy.testing.assert_array_almost_equal(g_mm,\
                numpy.array([[-0.00070252,  0.01052575,  0.00447283],
                             [ 0.00022631,  0.00010745,  0.00057586]]), 6)

if __name__ == "__main__":
    print("Full Tests for CNEO-PBC-QMMM.")
    unittest.main()
