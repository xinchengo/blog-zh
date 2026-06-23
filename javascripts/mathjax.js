window.MathJax = {
  loader: {load: [
    '[tex]/autoload', 
    '[tex]/ams',
    '[tex]/mathtools',
  ]},
  tex: {
    packages: {'[+]': 
      [
        'autoload',
        'ams',
        'mathtools',
      ]},
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  output: {
    font: 'mathjax-fira',
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
