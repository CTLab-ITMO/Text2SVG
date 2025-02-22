export default {
  multipass: true,
  js2svg: {
    indent: 0,
    pretty: true, 
  },
  plugins: [
    { name: 'cleanupAttrs' },
    { name: 'removeDoctype' },
    { name: 'removeXMLProcInst' },
    { name: 'removeComments' },
    { name: 'removeMetadata' },
    { name: 'removeEditorsNSData' },

    {
      name: 'cleanupNumericValues',
      params: {
        floatPrecision: 1,
        leadingZero: true,
        defaultPx: true,
        convertToPx: true
      },
    },
    {
      name: 'cleanupListOfValues',
      params: {
        floatPrecision: 0,
        leadingZero: true,
        defaultPx: true,
        convertToPx: true
      },
    },
    {
      name: 'cleanupIds',
      params: {
        remove: true,
      },
    },
    {
      name: 'convertColors',
      params: {
        names2hex: true,
        shorthex: true,
        rgb2hex: true,
        convertCase: 'upper',
      },
    },
    { name: 'removeNonInheritableGroupAttrs' },
    { name: 'removeUnusedNS' },
    { name: 'cleanupEnableBackground' },
    { name: 'convertStyleToAttrs' },
    {
      name: 'convertTransform',
      params: {
        floatPrecision: 0,
        convertToShorts: true,
        degPrecision: true,
        removeUseless: true,
        transformPrecision: 0,
        matrixToTransform: true,
        shortTranslate: true,
        shortScale: true,
        shortRotate: true,
        collapseIntoOne: true
      },
    },
    {
      name: 'convertPathData',
      params: {
        floatPrecision: 0,
        applyTransforms: true,
        applyTransformsStroked: true,
        makeArcs: true,
        straightCurves: true,
        convertToQ: true,
        convertToZ: true,
        curveSmoothShorthands: true,
        floatPrecision: 0,
        smartArcRounding: true,
        removeUseless: true,
        collapseRepeated: true,
        forceAbsolutePath: true
      },
    },
    {
      name: 'convertEllipseToCircle',
      params: {
        floatPrecision: 0,
      },
    },
    {
      name: 'convertShapeToPath',
      params: {
        floatPrecision: 0,
      },
    },
    { name: 'convertOneStopGradients' },
    { name: 'removeRasterImages' },
    { name: 'mergePaths' },
    { name: 'mergeStyles'},
    { name: 'collapseGroups' },
    { name: 'removeEmptyContainers' },
    { name: 'removeEmptyText' },
    { name: 'minifyStyles' },
    {
      name: 'inlineStyles',
      params: {
        onlyMatchedOnce: false,
      },
    },
    { name: 'moveElemsAttrsToGroup' },
    { name: 'moveGroupAttrsToElems' },
    { name: 'sortAttrs' },
    { name: 'reusePaths' },
    {
      name: 'removeAttrs',
      params: {
        attrs: '(version|xmlns|aria-hidden|class)',
      },
    },
    {
      name: 'cleanupIds',
      params: {
        remove: true,
        minify: true,
      },
    },
      
    { name: 'removeXlink', active: true },
    { name: 'removeUselessDefs', active: true },
    { name: 'removeDimensions', active: true },
    { name: 'removeViewBox', active: false }, // Keep viewBox for scaling
    { name: 'removeHiddenElems', active: true }, // Retain hidden elements
    { name: 'removeTitle', active: false },
    { name: 'removeDesc', active: false },
    { name: 'removeStyleElement', active: false },
    { name: 'removeScriptElement', active: false }
  ],
};
