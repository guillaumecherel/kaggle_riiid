{ pkgs ? import <nixpkgs> {} }:
  pkgs.mkShell {
    buildInputs = [ 
      pkgs.python37Packages.ipython 
      pkgs.python37Packages.matplotlib 
      pkgs.python37Packages.numpy 
      pkgs.python37Packages.pandas 
      pkgs.python37Packages.pyarrow 
      pkgs.python37Packages.pydot 
      pkgs.python37Packages.tensorflow_2
      pkgs.python37Packages.tensorflow-tensorboard_2
      pkgs.graphviz
      ];
    shellHook=''
      fish
    '';
}
