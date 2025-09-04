document.addEventListener('DOMContentLoaded', function() {
  console.log('Language switcher ready');

  // Add copy buttons to all code blocks
  document.querySelectorAll('pre').forEach(pre => {
    const code = pre.querySelector('code');
    if (!code) return;

    // Create the copy button
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.type = 'button';
    button.textContent = 'Copy';

    // Copy code to clipboard on click
    button.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.innerText);
        button.textContent = 'Copied!';
        setTimeout(() => (button.textContent = 'Copy'), 2000);
      } catch (err) {
        console.error('Failed to copy', err);
      }
    });

    pre.style.position = 'relative';
    pre.appendChild(button);
  });
});
