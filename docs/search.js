const PAGES = [
  { url: 'faq.html', title: 'Preguntas frecuentes' },
  { url: 'index.html', title: 'InsideForest v1.0' },
  { url: 'license.html', title: 'Licencia' },
  { url: 'benchmarks.html', title: 'Benchmarks' },
  { url: 'troubleshooting.html', title: 'Solución de problemas' },
  { url: 'roadmap.html', title: 'Roadmap' },
  { url: 'resources.html', title: 'Recursos externos' },
  { url: 'glossary.html', title: 'Glosario' },
  { url: 'changelog.html', title: 'Notas de versión' },
  { url: 'paper.html', title: 'Paper técnico de InsideForest' },
  { url: 'examples/index.html', title: 'Casos de uso' },
  { url: 'examples/marketing.html', title: 'Ejemplo: Segmentación de clientes' },
  { url: 'api/index.html', title: 'Referencia de API' },
  { url: 'guides/index.html', title: 'Guías de uso' },
  { url: 'guides/architecture.html', title: 'Guía: Arquitectura interna' },
  { url: 'guides/config.html', title: 'Guía: Configuración y parámetros' },
  { url: 'guides/interpretation.html', title: 'Guía: Interpretación de resultados' },
  { url: 'tutorials/index.html', title: 'Tutoriales' },
  { url: 'tutorials/persistence.html', title: 'Tutorial: Guardado y carga de modelos' },
  { url: 'tutorials/classification.html', title: 'Tutorial: Clasificación supervisada básica' },
  { url: 'tutorials/pipeline.html', title: 'Tutorial: Integración con pipelines de scikit-learn' },
  { url: 'tutorials/regression.html', title: 'Tutorial: Regresión supervisada' },
  { url: 'development/index.html', title: 'Sección de desarrollo' },
  { url: 'blog/index.html', title: 'Blog / Noticias' },
  { url: 'contributing/index.html', title: 'Contribuir' },
  { url: 'community/index.html', title: 'Comunidad' },
  { url: 'getting-started/index.html', title: 'Guía de inicio rápido' },
  { url: 'advanced/index.html', title: 'Recetas avanzadas' },
  { url: 'advanced/scaling.html', title: 'Receta avanzada: Uso en conjuntos de datos masivos' },
  { url: 'advanced/hyperparameter.html', title: 'Receta avanzada: Ajuste fino de hiperparámetros' }
];

function searchDocs(query) {
  const results = document.getElementById('search-results');
  results.innerHTML = '';
  if (!query) return;
  const q = query.toLowerCase();
  PAGES.forEach(page => {
    fetch(page.url)
      .then(resp => resp.text())
      .then(text => {
        if (text.toLowerCase().includes(q)) {
          const li = document.createElement('li');
          const a = document.createElement('a');
          a.href = page.url;
          a.textContent = page.title;
          li.appendChild(a);
          results.appendChild(li);
        }
      })
      .catch(err => console.error(err));
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('page-search');
  if (input) {
    input.addEventListener('input', () => searchDocs(input.value));
  }
});
