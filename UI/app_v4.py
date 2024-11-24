import numpy as np
import tifffile
from pathlib import Path
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from typing import List
import torch
import glob
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QMessageBox, QVBoxLayout
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer
from PyQt5.QtGui import QPainter, QImage, QPen, QColor, QPixmap, QFont, QBrush, QTransform
import math

class HelpOverlay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QLabel { background-color: rgba(0, 0, 0, 180); color: white; }")
        self.setText("""
        Billentyűparancsok:
        
        SPACE: Súgó mutatása/elrejtése
        ESC: Kilépés
        S: Mentés
        P: Pozitív pont mód
        N: Negatív pont mód
        M: Több kattintás mód
        A: Szegmens hozzáadása
        C: Megfeleltetések mutatása
        R: Tükrözés X tengely mentén
        SHIFT+R: Tükrözés Y tengely mentén
        DELETE/BACKSPACE: Kijelölt szegmens törlése
        H: Kimenet mutatása
        D: Kijelölt szegmens duplikálása
        U: Feltöltés
        """)
        self.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)
        self.hide()

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setMouseTracking(True)
        self.last_pos = QPoint()
        self.help_overlay = HelpOverlay(self)
        self.help_overlay.hide()
        self.qimage = None
        self.qimage_array = None
        self.original_qimage_array = None  # Eredeti kép tárolása
        self.setFocusPolicy(Qt.StrongFocus)  # Billentyűzet események fogadásához
        self.overlay_mask = None  # Szegmentálási maszk tárolása
        self.help_text = """
        Billentyűk:
        - Fel/Le: Sprite méretezése
        - Balra/Jobbra: Sprite forgatása
        - Delete: Sprite törlése
        - D: Kijelölt sprite duplikálása
        - R: Vízszintes tükrözés
        - Shift+R: Függőleges tükrözés
        
        Egér:
        - Bal klikk: Szegmentálás / Sprite kijelölés
        - Jobb klikk: Negatív pont hozzáadása
        - Húzás: Sprite mozgatása
        """
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.help_overlay:
            self.help_overlay.setGeometry(self.rect())

    def update_image(self, image):
        if image is not None:
            height, width = image.shape[:2]
            bytes_per_line = width * 4
            self.qimage_array = image
            self.original_qimage_array = image.copy()  # Eredeti kép mentése
            self.qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            self.update()

    def paintEvent(self, event):
        """Rajzolás"""
        if not self.qimage:
            return

        painter = QPainter(self)
        
        # Háttérkép rajzolása az eredeti képből
        if hasattr(self.parent, 'original_qimage_array'):
            qimage = QImage(
                self.parent.original_qimage_array.data,
                self.parent.original_qimage_array.shape[1],
                self.parent.original_qimage_array.shape[0],
                self.parent.original_qimage_array.strides[0],
                QImage.Format_RGBA8888
            )
            painter.drawImage(0, 0, qimage)
        else:
            painter.drawImage(0, 0, self.qimage)

        # Sprite-ok rajzolása
        for sprite in self.parent.sprites:
            # Transzformáció beállítása
            transform = QTransform()
            
            # Pozíció beállítása
            transform.translate(sprite.x, sprite.y)
            
            # Forgatás a középpont körül
            if sprite.angle != 0:
                transform.translate(sprite.qpixmap.width()/2, sprite.qpixmap.height()/2)
                transform.rotate(sprite.angle)
                transform.translate(-sprite.qpixmap.width()/2, -sprite.qpixmap.height()/2)
            
            # Méretezés a középpont körül
            if sprite.scale != 1.0:
                transform.translate(sprite.qpixmap.width()/2, sprite.qpixmap.height()/2)
                transform.scale(sprite.scale, sprite.scale)
                transform.translate(-sprite.qpixmap.width()/2, -sprite.qpixmap.height()/2)
            
            # Sprite rajzolása transzformációval
            painter.setTransform(transform)
            painter.drawPixmap(0, 0, sprite.qpixmap)
            
            # Kijelölés/hover keret rajzolása
            if sprite.selected or sprite.hover:
                pen = QPen(QColor(255, 255, 0) if sprite.selected else QColor(0, 255, 0))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawRect(0, 0, sprite.qpixmap.width(), sprite.qpixmap.height())
        
        # Pontok rajzolása
        if hasattr(self.parent, 'pos_points') and self.parent.pos_points:
            painter.setPen(QPen(QColor(0, 255, 0), 5))
            for x, y in self.parent.pos_points:
                painter.drawPoint(int(x), int(y))
                
        if hasattr(self.parent, 'neg_points') and self.parent.neg_points:
            painter.setPen(QPen(QColor(255, 0, 0), 5))
            for x, y in self.parent.neg_points:
                painter.drawPoint(int(x), int(y))

        self.help_overlay.raise_()

    def mouseMoveEvent(self, event):
        """Egér mozgatás kezelése"""
        x, y = event.pos().x(), event.pos().y()
        
        # Kiválasztott sprite mozgatása
        if event.buttons() & Qt.LeftButton:
            for sprite in self.parent.sprites:
                if sprite.selected:
                    sprite.drag_to(x, y)
        else:
            # Sprite-ok hover állapotának frissítése
            for sprite in self.parent.sprites:
                if sprite.intersect(x, y):
                    sprite.set_hover(True)
                else:
                    sprite.set_hover(False)
        
        self.last_pos = event.pos()
        self.update()

    def mousePressEvent(self, event):
        """Egér kattintás kezelése"""
        x, y = event.pos().x(), event.pos().y()
        
        # Ha van aktív sprite, csak azzal lehet interakcióba lépni
        active_sprite = next((sprite for sprite in self.parent.sprites if sprite.selected), None)
        
        if active_sprite:
            # Sprite területének ellenőrzése
            if active_sprite.intersect(x, y):
                active_sprite.start_drag(x, y)
        else:
            # Ha nincs aktív sprite, új sprite létrehozása vagy sprite kiválasztása
            clicked_sprite = None
            for sprite in reversed(self.parent.sprites):
                if sprite.intersect(x, y):
                    clicked_sprite = sprite
                    break
        
            if clicked_sprite:
                # Sprite kiválasztása
                clicked_sprite.set_selected(True)
            else:
                # Új sprite létrehozása csak akkor, ha nincs aktív sprite
                if self.parent.multi_click:
                    if self.parent.mode == 'positive':
                        self.parent.pos_points.append((x, y))
                    else:
                        self.parent.neg_points.append((x, y))
                else:
                    self.parent.pos_points = [(x, y)]
                    # Az _extract_segment a CollageApp osztályban van
                    if hasattr(self.parent, '_extract_segment'):
                        self.parent._extract_segment(self.parent.pos_points, self.parent.neg_points)
                    self.parent.pos_points = []
                    self.parent.neg_points = []
    
        self.last_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        """Egér felengedés kezelése"""
        if event.button() == Qt.LeftButton:
            # Forgatás és átméretezés befejezése
            for sprite in self.parent.sprites:
                sprite.set_hover(False)

    def keyPressEvent(self, event):
        """Billentyű események kezelése"""
        active_sprite = None
        for sprite in self.parent.sprites:
            if sprite.selected:
                active_sprite = sprite
                break
                
        if active_sprite:
            # Méretezés fel/le nyilakkal (2%-os lépésköz)
            if event.key() == Qt.Key_Up:
                active_sprite.scale_sprite(1.02)
            elif event.key() == Qt.Key_Down:
                active_sprite.scale_sprite(0.98)
            # Forgatás balra/jobbra nyilakkal
            elif event.key() == Qt.Key_Left:
                active_sprite.rotate(-5)
            elif event.key() == Qt.Key_Right:
                active_sprite.rotate(5)
            # Sprite törlése
            elif event.key() == Qt.Key_Delete:
                self.parent.sprites.remove(active_sprite)
                
            self.update()
        
        # Súgó megjelenítése/elrejtése
        if event.key() == Qt.Key_Space:
            if self.help_overlay.isVisible():
                self.help_overlay.hide()
            else:
                self.help_overlay.show()
            event.accept()
            return
            
        super().keyPressEvent(event)

class Sprite:
    def __init__(self, qpixmap, bbox=None):
        """
        Sprite osztály a kivágott képrészletek kezeléséhez
        
        Args:
            qpixmap: QPixmap objektum
            bbox: (x, y, w, h) befoglaló téglalap koordinátái
        """
        self.qpixmap = qpixmap
        self.bbox = bbox
        self.x = 0
        self.y = 0
        self.angle = 0
        self.scale = 1.0
        self.selected = False
        self.mask = None
        self.hover = False
        self.drag_start_x = 0
        self.drag_start_y = 0

    def get_center(self):
        """Visszaadja a sprite középpontját"""
        return QPoint(
            self.x + self.qpixmap.width() / 2,
            self.y + self.qpixmap.height() / 2
        )
        
    def intersect(self, x, y):
        """Ellenőrzi, hogy egy pont a sprite-on belül van-e"""
        return (self.x <= x <= self.x + self.qpixmap.width() and
                self.y <= y <= self.y + self.qpixmap.height())

    def set_hover(self, status: bool):
        """Beállítja a hover állapotot"""
        self.hover = status

    def set_selected(self, selected):
        self.selected = selected

    def move(self, dx, dy):
        self.x += int(dx)
        self.y += int(dy)

    def rotate(self, angle):
        self.angle += angle
        # Normalizáljuk 0-360 fokra
        self.angle = self.angle % 360

    def scale_sprite(self, factor):
        """Sprite méretezése egy szorzófaktorral"""
        new_scale = self.scale * factor
        # Korlátozzuk a méretezést, hogy ne legyen túl nagy vagy túl kicsi
        if 0.1 <= new_scale <= 5.0:
            self.scale = new_scale

    def start_drag(self, x, y):
        """Húzás kezdése - elmenti az egér relatív pozícióját a sprite-hoz képest"""
        self.drag_start_x = x - self.x
        self.drag_start_y = y - self.y

    def drag_to(self, x, y):
        """Húzás új pozícióra az egér pozíciója alapján"""
        self.x = int(x - self.drag_start_x)
        self.y = int(y - self.drag_start_y)

    def flipx(self):
        """Sprite vízszintes tükrözése"""
        # QPixmap -> QImage konvertálás
        image = self.qpixmap.toImage()
        # Tükrözés
        mirrored = image.mirrored(True, False)
        # QImage -> QPixmap konvertálás
        self.qpixmap = QPixmap.fromImage(mirrored)

    def flipy(self):
        """Sprite függőleges tükrözése"""
        # QPixmap -> QImage konvertálás
        image = self.qpixmap.toImage()
        # Tükrözés
        mirrored = image.mirrored(False, True)
        # QImage -> QPixmap konvertálás
        self.qpixmap = QPixmap.fromImage(mirrored)

class CollageApp(QMainWindow):
    def __init__(
        self,
        image_path: str = "data/lizards.jpg",
        output_folder="output",
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        model_type="vit_h",
        checkpoint="data/sam_vit_h_4b8939.pth",
    ):
        super().__init__()
        
        # Fájl és mappa kezelés
        self.filename = Path(image_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.save_counter = 1
        
        # GUI elemek inicializálása
        self.imageWidget = ImageWidget(self)
        self.setCentralWidget(self.imageWidget)
        self.setWindowTitle('MagicFixup')
        
        # Sprite-ok listája
        self.sprites = []
        
        # Pontok és állapot inicializálása
        self.pos_points = []
        self.neg_points = []
        self.mode = 'positive'  # Alapértelmezett mód: pozitív pontok
        self.multi_click = False  # Több kattintásos mód alapértelmezetten kikapcsolva
        
        # SAM modell betöltése
        print("Loading SAM...")
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        print(f"SAM loaded on: {device}")
        print("done.")
        
        # Kép betöltése
        self.load_image(image_path)
        
        # Súgó megjelenítése induláskor
        self.imageWidget.help_overlay.show()

        # Mentés számláló inicializálása
        edit_paths = glob.glob(str(self.output_folder / (self.filename.stem + '__edit__*png')))
        self.save_counter = len(edit_paths) + 1

    def load_image(self, image_path):
        """Kép betöltése"""
        # Eredeti kép betöltése BGR formátumban
        image = cv2.imread(image_path)
        if image is None:
            print(f"Hiba: nem sikerült betölteni a képet: {image_path}")
            return False

        # Kép méretezése, ha túl nagy
        max_size = 1600
        height, width = image.shape[:2]
        if width > max_size:
            scale = max_size / width
            width = max_size
            height = int(height * scale)
            image = cv2.resize(image, (width, height))

        # BGR -> RGB konvertálás
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Másolatok készítése
        self.original_image = image.copy()  # RGB formátumban tároljuk
        
        # Alfa csatorna hozzáadása
        alpha = np.ones((height, width, 1), dtype=np.uint8) * 255
        image_rgba = np.concatenate([image, alpha], axis=-1)
        
        # Frissítjük az ImageWidget-et
        self.imageWidget.update_image(image_rgba)
        
        # Ablak méretének beállítása
        self._resize_window()
        
        return True

    def _resize_window(self):
        # Ablak méretének beállítása
        scaled_width = min(1600, self.imageWidget.qimage_array.shape[1])  # Maximum 1600 pixel széles
        scale_factor = scaled_width / self.imageWidget.qimage_array.shape[1]
        scaled_height = int(self.imageWidget.qimage_array.shape[0] * scale_factor)
        
        # Központosítás a képernyőn
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - scaled_width) // 2
        y = (screen.height() - scaled_height) // 2
        
        self.setGeometry(x, y, scaled_width, scaled_height)

    def keyPressEvent(self, event):
        """Billentyű események kezelése"""
        # A szóköz kezelését átadjuk az ImageWidget-nek
        if event.key() == Qt.Key_Space:
            return
            
        if event.key() == Qt.Key_Escape:
            # Kilépés
            self.close()
        
        elif event.key() == Qt.Key_S:
            # Mentés
            self._save()
        
        elif event.key() == Qt.Key_P:
            # Pozitív pont mód
            self.mode = 'positive'
            print("Mode: positive")
        
        elif event.key() == Qt.Key_N:
            # Negatív pont mód
            self.mode = 'negative'
            print("Mode: negative")
        
        elif event.key() == Qt.Key_M:
            # Multi-click mód váltása
            self.multi_click = not self.multi_click
            print(f"Multi-click mode: {'on' if self.multi_click else 'off'}")
            # Üzenet megjelenítése
            if self.multi_click:
                self._show_message("Több kattintásos mód bekapcsolva", "rgba(0, 100, 200, 200)")
            else:
                self._show_message("Több kattintásos mód kikapcsolva", "rgba(100, 100, 100, 200)")
        
        elif event.key() == Qt.Key_A and self.multi_click:
            # Segmentálás multi-click módban
            if self.pos_points or self.neg_points:
                self._extract_segment(self.pos_points, self.neg_points)
                self.pos_points = []
                self.neg_points = []
        
        elif event.key() == Qt.Key_R:
            # Sprite tükrözése
            if len(self.sprites) > 0:
                selected_sprite = None
                for sprite in self.sprites:
                    if sprite.selected:
                        selected_sprite = sprite
                        break
                
                if selected_sprite:
                    if event.modifiers() & Qt.ShiftModifier:
                        # Shift+R: függőleges tükrözés
                        selected_sprite.flipy()
                        self._show_message("Sprite függőlegesen tükrözve", "rgba(0, 100, 200, 200)")
                    else:
                        # R: vízszintes tükrözés
                        selected_sprite.flipx()
                        self._show_message("Sprite vízszintesen tükrözve", "rgba(0, 100, 200, 200)")
                    self.imageWidget.update()
                else:
                    self._show_message("Nincs kijelölt sprite!", "rgba(200, 100, 0, 200)")
        elif event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            # Törlés
            for sprite in self.sprites[:]:  # Create a copy of the list to safely modify during iteration
                if sprite.selected:
                    self._delete_sprite(sprite)
        
        elif event.key() == Qt.Key_D:
            # Duplikálás
            self._duplicate_selected()
        
        elif event.key() == Qt.Key_U:
            # Feltöltés
            self._upload()
        
        # Frissítjük a megjelenítést
        self.imageWidget.update()

    def closeEvent(self, event):
        """Program bezárása"""
        print("Closing application...")
        # Cleanup
        self.predictor = None
        self.sam = None
        # Accept the close event
        event.accept()
        # Ensure the application exits
        QApplication.quit()

    def _extract_segment(self, pos_points, neg_points):
        """Szegmens kivágása és sprite létrehozása"""
        if not pos_points and not neg_points:
            return

        # Szegmentálás
        mask = self._segment(pos_points, neg_points)
        if mask is None:
            return
            
        # Sprite létrehozása a maszkból
        sprite = self._create_sprite_from_mask(mask)
        if sprite:
            # Töröljük ki a sprite területét az eredeti képből
            if sprite.bbox is not None:
                x, y, w, h = sprite.bbox
                
                # Eredeti kép módosítása - átlátszóvá tesszük a kivágott részt
                mask_area = mask[y:y+h, x:x+w]
                # Konvertáljuk a maszkot 0-1 közötti float értékekké
                mask_area = mask_area.astype(np.float32)
                # Alkalmazzuk a maszkot az alpha csatornára
                self.imageWidget.qimage_array[y:y+h, x:x+w, 3] = (self.imageWidget.qimage_array[y:y+h, x:x+w, 3] * (1 - mask_area)).astype(np.uint8)
                
                # QImage frissítése
                self.imageWidget.qimage = QImage(
                    self.imageWidget.qimage_array.data,
                    self.imageWidget.qimage_array.shape[1],
                    self.imageWidget.qimage_array.shape[0],
                    self.imageWidget.qimage_array.strides[0],
                    QImage.Format_RGBA8888
                )
                
            self.sprites.append(sprite)
            # Kijelöljük az új sprite-ot
            for s in self.sprites:
                s.selected = (s == sprite)
    
    def _segment(self, pos_points, neg_points):
        """Szegmentálás végrehajtása"""
        print("pos points", pos_points)
        print("neg points", neg_points)

        # Convert points to numpy arrays
        input_points = np.array(pos_points + neg_points)
        if len(input_points) == 0:
            return None

        input_labels = np.array([1] * len(pos_points) + [0] * len(neg_points))
        print("input points", input_points)
        print("labels", input_labels)

        # Set input image - már RGB formátumban van
        print("Setting SAM input image...")
        self.predictor.set_image(self.original_image)
        print("done")

        # Generate mask
        print("Running SAM...")
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        print("masks shape", masks.shape)
        print("done")

        # Return the mask with highest score
        mask_idx = np.argmax(scores)
        return masks[mask_idx]

    def _create_sprite_from_mask(self, mask):
        """Create a sprite from a mask."""
        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the mask and image to the bounding box
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
        cropped_arr = self.imageWidget.qimage_array[rmin:rmax+1, cmin:cmax+1]

        # Create sprite image with alpha channel
        sprite_img = np.zeros((cropped_mask.shape[0], cropped_mask.shape[1], 4), dtype=np.uint8)
        sprite_img[..., :3] = cropped_arr[..., :3]
        sprite_img[..., 3] = cropped_mask * 255

        # Create QImage from sprite
        qimg = QImage(sprite_img.data, sprite_img.shape[1], sprite_img.shape[0], 
                     sprite_img.strides[0], QImage.Format_RGBA8888)
        qpixmap = QPixmap.fromImage(qimg)
        
        # Create sprite
        sprite = Sprite(qpixmap, bbox=(0, 0, cropped_mask.shape[1], cropped_mask.shape[0]))
        sprite.x = cmin
        sprite.y = rmin
        sprite.mask = mask  # Store the full mask
        sprite.image_bbox = (rmin, rmax, cmin, cmax)  # Store the full image coordinates

        # Update original image - make the cut out region transparent
        alpha = self.imageWidget.qimage_array[rmin:rmax+1, cmin:cmax+1, 3].astype(np.float32)
        mask_float = (1 - cropped_mask).astype(np.float32)
        self.imageWidget.qimage_array[rmin:rmax+1, cmin:cmax+1, 3] = (alpha * mask_float).astype(np.uint8)
        
        # Update QImage with the modified array
        self.imageWidget.qimage = QImage(self.imageWidget.qimage_array.data, self.imageWidget.qimage_array.shape[1], 
                           self.imageWidget.qimage_array.shape[0], self.imageWidget.qimage_array.strides[0],
                           QImage.Format_RGBA8888)

        return sprite

    def _delete_sprite(self, sprite):
        """Sprite törlése"""
        if sprite in self.sprites:
            self.sprites.remove(sprite)
            # Visszaállítjuk az eredeti képet
            self.imageWidget.qimage_array = self.imageWidget.original_qimage_array.copy()
            
            # Újra kitöröljük a megmaradt sprite-ok területét
            for s in self.sprites:
                if s.bbox is not None:
                    x, y, w, h = s.bbox
                    mask = s.mask
                    if mask is not None:
                        mask_area = mask[y:y+h, x:x+w]
                        # Konvertáljuk a maszkot 0-1 közötti float értékekké
                        mask_area = mask_area.astype(np.float32)
                        # Alkalmazzuk a maszkot az alpha csatornára
                        self.imageWidget.qimage_array[y:y+h, x:x+w, 3] = (self.imageWidget.qimage_array[y:y+h, x:x+w, 3] * (1 - mask_area)).astype(np.uint8)
            
            # QImage frissítése
            self.imageWidget.qimage = QImage(
                self.imageWidget.qimage_array.data,
                self.imageWidget.qimage_array.shape[1],
                self.imageWidget.qimage_array.shape[0],
                self.imageWidget.qimage_array.strides[0],
                QImage.Format_RGBA8888
            )
            
            # Frissítjük a megjelenítést
            self.imageWidget.update()

    def _duplicate_selected(self):
        """Kijelölt sprite duplikálása"""
        for sprite in self.sprites:
            if sprite.selected:
                # Create new sprite with the same bbox
                new_sprite = Sprite(sprite.qpixmap, sprite.bbox)
                new_sprite.x = sprite.x + 20  # Offset slightly
                new_sprite.y = sprite.y + 20
                new_sprite.selected = True
                new_sprite.angle = sprite.angle
                new_sprite.scale = sprite.scale
                new_sprite.mask = sprite.mask
                new_sprite.image_bbox = sprite.image_bbox
                self.sprites.append(new_sprite)
        self.imageWidget.update()

    def _upload(self):
        """Kép feltöltése"""
        # Kiválasztunk egy képfájlt
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Kép kiválasztása",
            "",
            "Képfájlok (*.png *.jpg *.jpeg *.bmp *.tiff);;Minden fájl (*.*)"
        )
        
        if file_name:
            # Betöltjük a képet
            new_image = QImage(file_name)
            if not new_image.isNull():
                # Átméretezzük ha szükséges
                if new_image.size() != self.imageWidget.qimage.size():
                    new_image = new_image.scaled(
                        self.imageWidget.qimage.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                
                # Frissítjük a megjelenítést
                self.imageWidget.qimage = new_image
                self.imageWidget.update()
                print(f"Kép betöltve: {file_name}")
            else:
                print(f"Hiba a kép betöltésekor: {file_name}")

    def _save(self):
        """Aktuális kép mentése"""
        # Mentés előkészítése
        save_path = self.output_folder / f"{self.filename.stem}__edit__{self.save_counter:03d}.png"
        
        # Ideiglenes QImage létrehozása a teljes képernyő tartalmával
        screen = QImage(self.imageWidget.width(), self.imageWidget.height(), QImage.Format_RGBA8888)
        screen.fill(Qt.transparent)  # Átlátszó háttér
        
        # Painter a QImage-re
        painter = QPainter(screen)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Háttérkép rajzolása
        if self.imageWidget.qimage:
            painter.drawImage(0, 0, self.imageWidget.qimage)
        
        # Sprite-ok rajzolása
        for sprite in self.sprites:
            # Transzformáció beállítása
            transform = QTransform()
            transform.translate(sprite.x, sprite.y)
            
            # Forgatás a középpont körül
            if sprite.angle != 0:
                transform.translate(sprite.qpixmap.width()/2, sprite.qpixmap.height()/2)
                transform.rotate(sprite.angle)
                transform.translate(-sprite.qpixmap.width()/2, -sprite.qpixmap.height()/2)
            
            # Méretezés a középpont körül
            if sprite.scale != 1.0:
                transform.translate(sprite.qpixmap.width()/2, sprite.qpixmap.height()/2)
                transform.scale(sprite.scale, sprite.scale)
                transform.translate(-sprite.qpixmap.width()/2, -sprite.qpixmap.height()/2)
            
            # Sprite rajzolása transzformációval
            painter.setTransform(transform)
            painter.drawPixmap(0, 0, sprite.qpixmap)
        
        painter.end()
        
        # Kép mentése
        screen.save(str(save_path))
        
        # Sikeres mentés üzenet
        self._show_success_message()
        
        # Számláló növelése
        self.save_counter += 1

    def _show_message(self, text, color="rgba(0, 150, 0, 200)"):
        """Általános üzenet megjelenítése"""
        # Üzenet widget létrehozása
        message = QLabel(text, self)
        message.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
            }}
        """)
        message.adjustSize()
        
        # Üzenet középre pozicionálása
        message.move(
            (self.width() - message.width()) // 2,
            (self.height() - message.height()) // 2
        )
        
        # Üzenet megjelenítése
        message.show()
        
        # Időzítő a rejtéshez
        QTimer.singleShot(2000, message.deleteLater)  # 2 másodperc után eltűnik

    def _show_success_message(self):
        """Sikeres mentés üzenet megjelenítése"""
        self._show_message("Sikeres mentés!")

    def _hide_success_message(self):
        self.success_message_visible = False 
        self.imageWidget.update()

def main():
    app = QApplication(sys.argv)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/lizards.jpg"
    
    print("Launching app. Press `spacebar` for help.")
    import platform
    if platform.system() == 'Linux':
        import os
        os.environ['QT_XCB_GL_INTEGRATION'] = 'none'
    
    window = CollageApp(image_path=image_path)
    window.show()
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        print("Closing application...")

if __name__ == "__main__":
    main()
