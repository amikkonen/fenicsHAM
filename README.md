# fenicsHAM
A Heat Air Moisture (building physics) solver based on FEniCS

#######################################
# NOT READY! KNOWN ISSUES!
#######################################

In this project we solve heat and moisture transport in a solid, i.e. wall. The most challenging part is implementing the varying material properties. The validation case only has one material and works fine. You can either use the correaltion givein in standard SFS EN 15026, for example:

  self.kT = fe.Expression("""(1.5 + 15.8/1000*w)""",
                           w=self.w,
                           element = self.v.ufl_element())

or the more general method with tablulated values and interpolation. The interpolation is performed in interpolator.C using C. For a single material the interpolation works fine. FOR MULTIPLE MATERIALS THE MATERIAL PROPERTIES ARE WRONG! IT'S A BUG.

By default, the correlation form is used. You can change this by commenting/uncommenting the two lines.

    case.material_properties_equations()
#    case.material_properties_interpolation()

If you find the bug, please let me know! I have started using OpenFOAM for solving these kind of problems but it would be nice to fix this some day. 

