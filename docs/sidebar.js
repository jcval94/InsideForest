function filterNav() {
    const query = document.getElementById('search').value.toLowerCase();
    document.querySelectorAll('#nav-list li').forEach(li => {
        const text = li.textContent.toLowerCase();
        li.style.display = text.includes(query) ? '' : 'none';
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const search = document.getElementById('search');
    if (search) {
        search.addEventListener('input', filterNav);
    }
});
