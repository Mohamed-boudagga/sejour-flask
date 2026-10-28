(function () {
  "use strict";

  // --- Onglets -------------------------------------------------------------
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab"));
  function select(key) {
    tabs.forEach(function (t) {
      var on = t.dataset.target === key;
      t.setAttribute("aria-selected", on ? "true" : "false");
      document.getElementById("panel-" + t.dataset.target).hidden = !on;
    });
  }
  tabs.forEach(function (t, i) {
    t.addEventListener("click", function () { select(t.dataset.target); });
    t.addEventListener("keydown", function (e) {
      var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
      if (j === null) return;
      var n = tabs[(j + tabs.length) % tabs.length];
      n.focus(); select(n.dataset.target); e.preventDefault();
    });
  });

  // --- Formulaires ---------------------------------------------------------
  function fr(x) { return String(x).replace(".", ","); }
  function pct(days) { return ((days - 1) / 13) * 100; }

  document.querySelectorAll("form[data-scenario]").forEach(function (form) {
    var panel = form.closest(".panel");
    var card = panel.querySelector(".result-card");
    var box = card.querySelector(".result");
    var empty = card.querySelector(".result-empty");
    var apiErr = card.querySelector('[data-r="erreur"]');
    var btn = form.querySelector(".btn");
    function r(name) { return card.querySelector('[data-r="' + name + '"]'); }

    function clearErrors() {
      form.querySelectorAll(".field").forEach(function (f) { f.classList.remove("invalid"); });
      form.querySelectorAll(".err").forEach(function (e) { e.textContent = ""; });
    }
    function showErrors(champs) {
      var first = null;
      Object.keys(champs).forEach(function (name) {
        var span = form.querySelector('[data-err="' + name + '"]');
        if (!span) return;
        span.textContent = champs[name];
        span.closest(".field").classList.add("invalid");
        first = first || span.closest(".field").querySelector("input,select");
      });
      if (first) first.focus();
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      clearErrors();
      apiErr.hidden = true;
      var features = {};
      new FormData(form).forEach(function (v, k) { features[k] = v; });

      btn.disabled = true; btn.textContent = "Calcul en cours…";
      fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: form.dataset.scenario, features: features })
      }).then(function (res) {
        return res.json().then(function (data) { return { ok: res.ok, status: res.status, data: data }; });
      }).then(function (out) {
        if (!out.ok) {
          if (out.status === 422 && out.data.champs) showErrors(out.data.champs);
          empty.hidden = true; box.hidden = true;
          apiErr.textContent = (out.status === 422 && out.data.champs) ? "Corrigez les champs signalés en rouge, puis relancez l'estimation." : (out.data.erreur || "Une erreur est survenue.");
          apiErr.hidden = false;
          return;
        }
        var d = out.data;
        empty.hidden = true; apiErr.hidden = true; box.hidden = false;
        r("jours").textContent = fr(d.jours);
        var b = r("categorie");
        b.textContent = "Séjour " + d.categorie;
        b.className = "badge " + d.categorie;
        var lo = d.intervalle[0], hi = d.intervalle[1];
        r("range").style.left = pct(lo) + "%";
        r("range").style.width = (pct(hi) - pct(lo)) + "%";
        r("dot").style.left = pct(d.jours) + "%";
        r("intervalle").textContent = "de " + fr(lo) + " à " + fr(hi) + " jours";
        r("marge").textContent = fr(d.marge);
        r("fiabilite").textContent = d.fiabilite.texte + " (R² = " + fr(d.fiabilite.r2) + ")";
        var ul = r("avertissements"); ul.innerHTML = "";
        d.avertissements.forEach(function (w) {
          var li = document.createElement("li"); li.textContent = w; ul.appendChild(li);
        });
        if (window.matchMedia("(max-width: 820px)").matches) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }).catch(function () {
        empty.hidden = true; box.hidden = true;
        apiErr.textContent = "Impossible de joindre le serveur. Réessayez dans un instant.";
        apiErr.hidden = false;
      }).then(function () {
        btn.disabled = false; btn.textContent = "Estimer la durée";
      });
    });
  });
})();
