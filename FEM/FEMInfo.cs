﻿using Grasshopper;
using Grasshopper.Kernel;
using System;
using System.Drawing;

namespace FEM
{
    public class FEMInfo : GH_AssemblyInfo
    {
        public override string Name => "FEM";

        //Return a 24x24 pixel bitmap to represent this GHA library.
        public override Bitmap Icon => null;

        //Return a short string describing the purpose of this GHA library.
        public override string Description => "";

        public override Guid Id => new Guid("edee659d-c80e-4d16-8e3c-fbb69d711c08");

        //Return a string identifying you or your company.
        public override string AuthorName => "";

        //Return a string representing your preferred contact details.
        public override string AuthorContact => "";
    }
}