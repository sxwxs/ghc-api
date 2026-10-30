/* Shared top navigation for every ghc-api page.
 *
 * Each page used to hard-code its own link list, so the nav drifted: a page
 * linked to whatever existed when it was written, and new pages were missing
 * from older ones. The list lives here now, and a page only declares its title
 * plus any page-specific action buttons.
 *
 * Usage (place the script immediately after the placeholder so the bar is
 * rendered before the rest of the page's inline scripts run):
 *
 *   <nav class="navbar" id="app-nav" data-title="Request Browser">
 *       <template id="nav-extras">
 *           <button onclick="downloadRequests()">Download .jl</button>
 *       </template>
 *   </nav>
 *   <script src="/static/nav.js"></script>
 *
 * Optional attributes on the placeholder:
 *   data-active="/request-stats"  force which entry is highlighted (for detail
 *                                 pages that are not themselves in the list).
 */
(function () {
    'use strict';

    // Shown inline: the destinations worth one click.
    var PRIMARY = [
        { href: '/', label: 'Dashboard' },
        { href: '/live', label: 'Live Throughput' },
        { href: '/requests', label: 'Request Browser' },
        { href: '/chat', label: 'Chat' },
        { href: '/code-agent-manager', label: 'Code Agent Manager', badge: 'config-manager-alert' }
    ];

    // Behind the "Other" button: occasional tools and raw JSON views.
    var SECONDARY = [
        // Available everywhere: on pages that define the modal it opens it,
        // elsewhere the link lands on the dashboard which opens it from the hash.
        { href: '/#runtime-config', label: 'Runtime Config Override', action: 'openRuntimeConfigModal' },
        { href: '/agent', label: 'Code Agent' },
        { href: '/request-stats', label: 'Request Stats' },
        { separator: true },
        { href: '/api/stats', label: 'API Stats (JSON)' },
        { href: '/api/live-stats', label: 'Live Stats (JSON)' }
    ];

    function pathOf(href) {
        return href.split('#')[0].split('?')[0] || '/';
    }

    function normalize(path) {
        if (!path) {
            return '/';
        }
        // Treat /requests and /requests/ as the same entry.
        return path.length > 1 && path.charAt(path.length - 1) === '/'
            ? path.slice(0, -1)
            : path;
    }

    function createItem(item, activePath) {
        var element = document.createElement('a');
        element.href = item.href;
        element.textContent = item.label;

        // Items carrying a fragment are in-page actions (e.g. the runtime
        // config modal), not destinations, so they never claim to be current.
        if (item.href.indexOf('#') === -1 && normalize(pathOf(item.href)) === activePath) {
            element.setAttribute('aria-current', 'page');
        }

        if (item.badge) {
            var badge = document.createElement('span');
            badge.id = item.badge;
            badge.className = 'config-manager-alert';
            badge.title = 'Config differs from the shared copy';
            badge.textContent = '!';
            element.appendChild(badge);
        }

        if (item.action) {
            // Prefer the in-page action when this page implements it; otherwise
            // fall through to the href so the behavior still exists elsewhere.
            element.addEventListener('click', function (event) {
                var handler = window[item.action];
                if (typeof handler === 'function') {
                    event.preventDefault();
                    handler();
                }
            });
        }

        return element;
    }

    function buildMenu(activePath) {
        var wrapper = document.createElement('div');
        wrapper.className = 'nav-menu';

        var toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'nav-menu-toggle';
        toggle.setAttribute('aria-haspopup', 'true');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.innerHTML = 'Other <span class="nav-caret" aria-hidden="true">\u25be</span>';

        var panel = document.createElement('div');
        panel.className = 'nav-menu-panel';
        panel.setAttribute('role', 'menu');

        SECONDARY.forEach(function (item) {
            if (item.separator) {
                var separator = document.createElement('div');
                separator.className = 'nav-menu-sep';
                panel.appendChild(separator);
                return;
            }
            var entry = createItem(item, activePath);
            entry.setAttribute('role', 'menuitem');
            panel.appendChild(entry);
        });

        function close() {
            wrapper.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        }

        toggle.addEventListener('click', function (event) {
            event.stopPropagation();
            var open = wrapper.classList.toggle('open');
            toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        panel.addEventListener('click', function (event) {
            // Let the click reach the link, then collapse the menu.
            if (event.target.closest('a, button')) {
                close();
            }
        });
        document.addEventListener('click', function (event) {
            if (!wrapper.contains(event.target)) {
                close();
            }
        });
        document.addEventListener('keydown', function (event) {
            if (event.key === 'Escape') {
                close();
            }
        });

        wrapper.appendChild(toggle);
        wrapper.appendChild(panel);
        return wrapper;
    }

    function render(root) {
        var activePath = normalize(root.dataset.active || window.location.pathname);

        var extras = root.querySelector('#nav-extras');
        var extraNodes = extras ? Array.prototype.slice.call(extras.content.children) : [];

        root.classList.add('navbar');
        root.innerHTML = '';

        var heading = document.createElement('h1');
        heading.textContent = root.dataset.title || 'Copilot API Proxy';
        root.appendChild(heading);

        var actions = document.createElement('div');
        actions.className = 'navbar-actions';
        PRIMARY.forEach(function (item) {
            actions.appendChild(createItem(item, activePath));
        });
        // Page-specific controls sit between the shared links and "Other".
        extraNodes.forEach(function (node) {
            actions.appendChild(node);
        });
        actions.appendChild(buildMenu(activePath));
        root.appendChild(actions);

        // Full-height pages (chat, agent, request browser) size their body
        // against the bar. Publishing the measured height keeps them correct
        // even when the bar wraps to two rows on a narrow window.
        var publishHeight = function () {
            // Keep the stylesheet fallback when there is no layout to measure.
            if (root.offsetHeight > 0) {
                document.documentElement.style.setProperty(
                    '--nav-height', root.offsetHeight + 'px');
            }
        };
        publishHeight();
        window.addEventListener('resize', publishHeight);
    }

    var placeholder = document.getElementById('app-nav');
    if (placeholder) {
        render(placeholder);
    }
})();
