# adobe-magic-fixup

Adobe Magic Fixup

Tuning for Adobe Magic Fixup projekt: [https://magic-fixup.github.io/](https://magic-fixup.github.io/)

[GitHub Magic Fixup](https://github.com/adobe-research/MagicFixup)

Cseréld le a 3 fájlt a megfelelő helyen a GitHub-ról letöltött forráskódban.

[Tutorial video](https://youtu.be/x7A3XKlor8g)

## Generálás
- **15 GByte VRAM**
- Az új verzió **LANCZOS átméretezést** használ, ami sokkal jobb minőségű képeket eredményez.
- Debug információk a modell kimenetéről:
  - Könnyebb hibaelhárítás
  - Pontosabb színkezelés
- **PNG formátum** a jobb minőségért.
- Informatív címkék és leírások.
- Hatékonyabb tensor műveletek.
- Csak szükség esetén méretez át.

## Szerkesztő
### GPU Támogatás és Teljesítmény
- Az új verzió automatikusan felismeri és használja a **GPU-t**, ha elérhető.
- Explicit üzenet jelzi, hogy a program **GPU-n vagy CPU-n** fut.
- Jobb teljesítmény a GPU gyorsítás miatt.

### Felhasználói Visszajelzés
- Új **„Sikeres mentés!”** felirat jelenik meg 3 másodpercre a mentés után:
  - A felirat zöld háttérrel, középen jelenik meg.
- Azonnali vizuális visszajelzés a felhasználói műveletekről.

### Színkezelés Javítások
- **BGR-RGB konverzió** implementálása a képek betöltésénél és mentésénél.
- A képek most már helyes színekben jelennek meg.
- Az elmentett fájlok is megtartják az eredeti színeket.

### Súgó Rendszer Fejlesztése
- Magyar nyelvű, átláthatóbb súgó szöveg.
- Jobb strukturálás bullet pointokkal.
- Nagyobb betűméret a jobb olvashatóságért.
- Sárgás címsor kiemelés.
- Sötétebb, átlátszóbb háttér.

### Sprite Műveletek Fejlesztése
- Egyértelműbb sprite manipulációs lehetőségek.
- Részletes magyarázat a billentyűparancsokról.
- Új műveletek hozzáadása:
  - **Mozgatás**: Kattintás + húzás
  - **Forgatás**: CTRL + húzás
  - **Nagyítás**: SHIFT + húzás
  - **X-tengely nagyítás**: CTRL+SHIFT + húzás

### Billentyűparancsok Optimalizálása
- A **„P” gomb konfliktus** feloldása.
- Az **„F” gomb** használata a feldolgozáshoz.
- Logikusabb billentyűkiosztás.

### Technikai Fejlesztések
- **Típushibák javítása** (pl. `List[str]` helyett `list[str]`).
- Jobb hibakezelés.
- Kód tisztítás és optimalizálás.

### Dokumentáció
- Részletesebb kódkommentek.
- Magyar nyelvű üzenetek.
- Felhasználóbarát hibaüzenetek.

---

Ezek a fejlesztések együttesen egy sokkal felhasználóbarátabb, stabilabb és hatékonyabb alkalmazást eredményeztek. A program most már:
- Gyorsabban fut a **GPU támogatás** miatt.
- Könnyebb használni a jobb visszajelzések miatt.
- Stabilabb a javított színkezelés miatt.
- Intuitívabb a fejlesztett súgó rendszer miatt.
- Professzionálisabb megjelenésű a vizuális fejlesztések miatt.
